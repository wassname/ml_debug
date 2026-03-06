Source: https://old.reddit.com/r/reinforcementlearning/comments/bzg3l2/
Title: How to *more intelligently* debug RL roadblocks?
Fetched-via: Reddit JSON API (limit=500, depth=10)
Fetch-status: verbatim

# How to *more intelligently* debug RL roadblocks?

**Posted by:** u/GrundleMoof | Score: 4 | 7 comments

A while ago I [made this post](https://www.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/) asking for tips on debugging when you run into a problem with RL.

However, I think the majority of the advice can be summed up with:

1) Test bits individually to make sure they're doing what they should

2) Don't go down a rabbit hole of fiddling with hyperparameters

3) Log/record/display everything, and "look for things that are acting funny"


and I just want to be clear that I'm not disparaging that advice, it's actually really good, I'm thankful, and I know I'm asking a tricky, general question!

But I want to get to the "next level". I think I know the theory well enough, and I've successfully done a few toy problems, but I'm still here banging my head against the wall.

I'll take a practical example I'm struggling with now: gym's `Pendulum-v0`, which has a continuous action space of [-2, 2], and three state variables (`(cos(theta), sin(theta), theta_dot)`). I'm trying to solve it with a fairly simple AC setup and PyTorch. I'm using the RMSprop optimizer, and 2 (or 3) fully connected NN layers, with 50 (or 100) units in each layer, to approximate pi (the policy) and V (the value function/baseline).

To select the actions, like in [the A3C paper](https://arxiv.org/pdf/1602.01783.pdf), I have the pi NN have two outputs, mu and sd2 (the standard deviation squared). Every time step, I select an action `a` from a normal distribution with that mu and `sd**2`. Then, I calculate that `pi(a)` (just from the equation of a normal dist. with that mu, `sd**2`), and iterate the agent to get the reward from that time step.

Also like the A3C paper (for the Pendulum problem), I'm doing all the updates at once, at the end of each episode (so it's basically MC with V as the baseline). For each time step (after the episode) I accumulate the rewards from t to t_max as `r_accum` (with gamma = 0.99), then say `V_loss = (r_accum - V_list).pow(2).sum()`. For the policy gradient, I do `policy_loss = -(torch.log(pi_list)*(r_accum - V_list)).sum()`, and then zero grads, backwards the losses, step the optimizer, etc.

And I'm just not seeing any learning, going up to about 20k episodes. I'm plotting to TensorBoard (losses, rewards, weights, biases, gradients), but nothing is striking me as an obvious culprit. It gets varying rewards, the V_loss seems to decrease to 0, and the policy_loss usually kind of wanders but eventually goes to 0 (I think because it's also proportional to (r_accum - V_list) which is also going to 0).

But I think this is a perfect learning example. This is doable (...right?), it seems mostly correctly set up, and it's probably a fairly simple fix if I knew how to diagnose it. For the more experienced RL'ers out there, where would you start? What would you look at? What would you verify is working correctly?

Here are some of my guesses/notes:

* I haven't actually seen any straightforward implementations of a vanilla PG algo solving Pendulum-v0. In the A3C paper, they add an LSTM to it. There are a bunch of DDPG papers online, but that's a pretty different story. I found one A3C that doesn't seem to have an LSTM, so I'll check that out.
* Do I need experience replay? Maybe the variance is just too high using essentially REINFORCE with this problem, so I need to be getting much better data efficiency (or running it for a ton longer) ?
* I was worried that maybe it was never actually getting to positions where it could get a high enough reward (to "motivate" it to reach those positions), but I plotted some trajectories and it's definitely getting up to the top (by swinging wildly anyway), where R = 0, so it's definitely experiencing them.

Things I've tried (but maybe not systematically enough):

* Different initial LRs
* Different optimizers
* Different number of hidden layers/units
* Shared pi/V NN body (with diff output layers) vs not
* Changing amount of entropy
* Adding correlated noise
* Using TD residual instead of MC version
* Clipping the gradient
* Different gamma values

Anyway, I'd love it if anyone has any more general advice for how to think about and go about solving RL problems. I of course want to solve this one, but I want a more general way of thinking.

## Comments

**u/i_do_floss** (score: 3):
I dont have the answer for you, but I had an algorithm that was stuck on pendulum for a while and these eventually ended up being the issues:

1. The environment is wrapped with a wrapper that kills the environment after 200 steps. I was ignoring that so I could use 1024 steps. So I ignored the "done" / "is-terminal" variable but I forgot to exclude it from my stored memories in my memory buffer so my updates were all wrong.

I decided observing the predicted value and seeing if it was crazy (high variance) may be an indicator of an issue.

Also I could use tensorboard and visualize runtime information so I could see what was going into my placeholders.

2. My q value was shaped (none x 1), and my placeholders for rewards/terminals were shaped ( none) when I compared those in a tensor I ended up with a tensor shaped ( none , none) which didnt do what I expected

I decided I could mitigate that type of issue in the future by writing my expected shapes of the networks in a notebook and checking if they match afterward using tensorboard. Some people use assert shape functions.

Also, just so you know, I'm training soft actor critic in about 20 episodes of length 1024. I don't think you should wait for 1000s of episodes.
 
Pendulum v0 is an easy environment for your algorithm to learn. I suggest sticking with these hyper parameters. If they don't work, it's probably your algorithm.

policy network size: [64, 64]
batch size: 256
gamma: 0.99
adam optimizer
relu network activations (on every layer except the last one which has no activation)

Lastly, make sure your action space allows your algorithm to output actions in the space of -2 to 2.

  **u/GrundleMoof** (score: 1):
  &gt; The environment is wrapped with a wrapper that kills the environment after 200 steps. I was ignoring that so I could use 1024 steps. So I ignored the "done" / "is-terminal" variable but I forgot to exclude it from my stored memories in my memory buffer so my updates were all wrong.
  
  So I currently have my agent as a wrapper for the gym env, and it returns a tuple of (reward, state_next, done), and I break on done.
  
  &gt; I decided observing the predicted value and seeing if it was crazy (high variance) may be an indicator of an issue.
  
  Hmmm, by value, you mean the value function? And do you mean variance across different states, or the same state over time?
  
  &gt; Also I could use tensorboard and visualize runtime information so I could see what was going into my placeholders.
  &gt; 
  &gt; My q value was shaped (none x 1), and my placeholders for rewards/terminals were shaped ( none) when I compared those in a tensor I ended up with a tensor shaped ( none , none) which didnt do what I expected
  &gt; I decided I could mitigate that type of issue in the future by writing my expected shapes of the networks in a notebook and checking if they match afterward using tensorboard. Some people use assert shape functions.
  
  ahh yeah that's some good advice. I actually got burned by that earlier in this project, but figured it out by printing the sizes. PyTorch is a little tricky in that it will accept multiplying tensors of various combinations of sizes, with different results... so I should probably do asserts from now on.
  
  &gt; Also, just so you know, I'm training soft actor critic in about 20 episodes of length 1024. I don't think you should wait for 1000s of episodes.
  
  Hmm, so right now I'm trying a pretty simple setup, just a policy gradient with a value function. I don't know much about SAC, but it seems more advanced.
  
  I was starting to get skeptical whether this setup could even learn a continuous action space problem like Pendulum-v0, because when I searched for stuff, almost everything I found was using at least DDPG or more complex. But then I found [this guy's project](https://github.com/MorvanZhou/pytorch-A3C), just A3C, and it solves it pretty quickly and reliably.
  
  I started going through his code and it's nearly exactly the same as mine. I thought that it's possible that using 4 workers has a "decorrelating" effect (like experience replay), so I changed his code to drop it to 1 worker, and it still works! So it's clearly something else and I haven't figured it out yet. It's so similar to mine though, both in terms of setup and hyperparameters...
  
  &gt; Pendulum v0 is an easy environment for your algorithm to learn. I suggest sticking with these hyper parameters. If they don't work, it's probably your algorithm.
  &gt; 
  &gt; policy network size: [64, 64] batch size: 256 gamma: 0.99 adam optimizer relu network activations (on every layer except the last one which has no activation)
  
  You mean, two hidden layers of size 64 each? And are you outputting a value function too?
  
  So, maybe I'm missing something here -- do you mean batches of episodes, or batches of steps? I'm using gamma = 0.9 or 0.99. I've tried Adam and RMSprop, no success with either... I'm using tanh activations, but that probably shouldn't change anything significantly, right?
  
  &gt; Lastly, make sure your action space allows your algorithm to output actions in the space of -2 to 2.
  
  Yeah, my policy outputs a mu and sigma. The mu output is 2*tanh, so it's mapped to -2, 2, and the sigma one (actually sigma^2 ) is put through a softplus output.

    **u/i_do_floss** (score: 1):
    Yes two hidden layers with 64 nodes. The value function is a third layer basically.
    
    Tanh on last layer makes sense for policy. What are you using on hidden layers and value function final layer?
    
    Also, have you tried different reward scales?

      **u/GrundleMoof** (score: 1):
      Hi again, sorry for the delay! I was traveling with no service...
      
      I've tried a few different topologies. Right now I'm doing this:
      
      		self.actor_lin1 = nn.Linear(3, 200)
      		self.mu = nn.Linear(200, 1)
      		self.sigma = nn.Linear(200, 1)
      		self.critic_lin1 = nn.Linear(3, 100)
      		self.v = nn.Linear(100, 1)
      
      and for my forward():
      
      		y = torch.tanh(self.critic_lin1(x))
      		v = self.v(y)
      
      		z = torch.tanh(self.actor_lin1(x))
      		mu = 2*torch.tanh(self.mu(z))
      		sd2 = softplus(self.sigma(z)) + 0.001
                      return(v, (mu, sd2))
      
      So I'm using tanh() for the nonlinearities as well. I'm adding that 0.001 to the sd2 because it keeps it from getting too small (which should be enforced by the entropy term anyway) and I've seen it done in a few formulations of this.
      
      I also tried with having the mu/sigma layers combined into a nn.Linear(200, 2) layer (which should be functionally the same I think), as well as having the mu/sigma and v outputs share the first nn.Linear(3, 200) layer before splitting off (which is different, the shared head thing, but I've used elsewhere and seen people use).
      
      I'm scaling the rewards in a way I've seen a bunch of other people do. Since the reward each step has the range [-16, 0], I'm normalizing it by doing (r + 8.0)/8.0, which should put it about in the range [-1, 1].
      
      At this point I'm basically trying to replicate the guy's A3C implementation from above (minus the multiple workers part, but I ran his with 1 worker and it reliably improves every time). Mine *does* seem to improve, but really slowly compared to his, and sometimes seems to get worse after a while. Like, it's not not improving *at all*, just very slowly and also not reliably, which means something must be off.

        **u/i_do_floss** (score: 1):
        tanh activations are really sensitive to the weights and bias initialization. Is he using tanh activations?
        
        tanh makes sense to me for the actor output. But I would probably use relu for the nonlinearities so the initialization is easier.
        
        tanh starts to experience issues when the inputs and outputs are too big. (bigger than like -.6 and 6.)

      **u/GrundleMoof** (score: 1):
      by the way, just to give an example:
      
      [Here's an image of the reward per episode using his code](https://imgur.com/9K2clLs)
      
      [and here's mine](https://imgur.com/IMy6Rb5)
      
      Both with moving averages shown, to smooth it out.
      
      You can see that mine *does* improve, up to about episode 2000, but then gets worse. It pretty consistently does that. His on the other hand, always improves and stays good.
      
      To me, that indicates that it's almost there, but something's going on with the optimizer or something, like maybe it becomes unstable or something. But I'm using the same LR he is (2e-4), and I've tried both Adam (like him) and RMSprop.

**u/i_do_floss** (score: 3):
Also, this only applies to pendulum v0, but it's a great first environment for spinning up an algorithm because you can graph the entire state space on a 2 dimensional plane (as an image). I graph my policy / q assessments in polar coordinates where radius is the velocity and theta is the angle of the pendulum.

&amp;#x200B;

Here's a link to the code I use to do it.

[https://github.com/DanielSmithMichigan/reinforcement-learning/blob/72b0e939c47234fd63c8459fdfa35f18d9053b49/soft-actor-critic/agent/Agent.py#L287](https://github.com/DanielSmithMichigan/reinforcement-learning/blob/72b0e939c47234fd63c8459fdfa35f18d9053b49/soft-actor-critic/agent/Agent.py#L287)

&amp;#x200B;

Here is an album of screenshots I took while training a successful policy.

&amp;#x200B;

[https://imgur.com/a/05C5vVa](https://imgur.com/a/05C5vVa)

&amp;#x200B;

I hope that helps. Feel free to hit me up on discord if you want someone to talk through it with. I'm kind of in the same boat as you with regard to wishing I knew the better ways to debug RL algorithms.

&amp;#x200B;

(Discord: Perseus#5383)
