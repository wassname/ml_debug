Source: https://old.reddit.com/r/reinforcementlearning/comments/7s8px9/deep_reinforcement_learning_practical_tips/
Title: Deep Reinforcement Learning practical tips
Fetched-via: browser paste (user)
Fetch-status: verbatim

# Deep Reinforcement Learning practical tips

submitted 8 years ago by grupiotr | 14 points (90% upvoted) | 13 comments

I would be particularly grateful for pointers to things you don't seem to be able to find in papers. Examples include:

- How to choose learning rate?
- Problems that work surprisingly well with high learning rates
- Problems that require surprisingly low learning rates
- Unhealthy-looking learning curves and what to do about them
- Q estimators deciding to always give low scores to a subset of actions effectively limiting their search space
- How to choose decay rate depending on the problem?
- How to design reward function? Rescale? If so, linearly or non-linearly? Introduce/remove bias?
- What to do when learning seems very inconsistent between runs?
- In general, how to estimate how low one should be expecting the loss to get?
- How to tell whether my learning is too low and I'm learning very slowly or too high and loss cannot be decreased further?

## Comments

**u/wassname** (11 points):

Resources: I found these very useful

- Deep RL Bootcamp Lecture 6: Nuts and Bolts of Deep RL Experimentation (slides) and a written summary
- The 3 NIPS2017 Learning to run write ups contain practical advice from a competition
- Lessons Learned Reproducing a Deep Reinforcement Learning Paper
- Deep Reinforcement Learning that Matters - this gives you an idea of what does and doesn't matter
- Deep Reinforcement Learning Doesn't Work Yet (at least as well as the hype suggests)
- General deep learning tips from Slav Ivanov

Lessons learnt:

- log everything with tensorboard/tensorboardX: policy and critic losses, advantages, ratio, actions (mean and std), states, noise. Check values, check losses are decreasing etc.
- keep track of experiments with an experiments log (git commit messages with non-committed data or logs stored by date)
- clip and clamp: mistakes not obvious as they can cause values to blow up instead of NaN
  - clamp all values, logarithmic values: `logvalue.clamp(-np.log(1e-5), np.log(1e-5))`
  - watch out for dividing by a value: `1/std` should be `1/(std+eps)` where `eps=1e-5`
  - clip gradients: `grad_norm = torch.nn.utils.clip_grad(model.params, 20)`, then log grad norm
- normalise everything: use running norms for state and reward; layer norms help
- check everything: plot and sanity check as many values as possible. Check initial outputs, inits, distributions, action range.
- think about step-size/sampling-rate: RL is sensitive to it (action repeat, frame skipping). Papers found skipping 4 Atari frames helped, repeating 4 actions in "Learning to Run" helped.

Curves:

- in PPO the std should decrease as it learns
- in actor-critic the critic loss should start converging then the actor loss follows
- watch for local minima where it outputs a constant action
- watch gradients for actor and critic; if much lower than 20 or much larger than 100 often run into problems (20 and 40 are where projects often clip gradient norm)
- run on CartPole and log same curves to see what healthy looks like

Reward:

- It's not the scaling factor that matters but the final value. Papers have gotten good results with rewards between 100-1000.

Learning rate:

- Use decaying learning rates, watch loss curves to see when they begin to converge.
- loss_actor will often initially increase while the critic is doing its initial learning (value function is a moving target). Focus on making the critic learning rate work first.
- Critic learning rates are often set higher, with larger batches.
- Use cyclical learning rate trick: slowly increase LR to find the min where model learns and max where it still converges.

My own questions:

- How do you know if you've set exploration/variance too high or low?
- Should you use a multi-headed actor/critic? Or separate networks?

"What to do when learning seems very inconsistent between runs?" - This could be an init issue. Try to init so it defaults to reasonable action values even before training.

---

**u/gwern** (8 points):

I've seen similar engineering details & folklore, but mostly in slides/talks:
- https://www.reddit.com/r/reinforcementlearning/comments/6vcvu1/icml_2017_tutorial_slides_levine_finn_deep/
- https://www.reddit.com/r/reinforcementlearning/comments/75m5vd/deep_rl_bootcamp_2017_slides_and_talks/
- https://www.reddit.com/r/reinforcementlearning/comments/5i67zh/deep_reinforcement_learning_through_policy/
- https://www.reddit.com/r/reinforcementlearning/comments/5hereu/the_nuts_and_bolts_of_deep_rl_research_schulman/

  **u/twkillian** (1 point): I was about to post John Schulman's talk here as well. Great resource.

  **u/wassname** (1 point): Summarising the ones I hadn't seen:
  - 5i67zh: fix random seed to reduce variance; think about step-size/sampling-rate; RL sensitive to optimizer choice (SGD, Adam)
  - 6vcvu1: slides focused more on algorithm choice/design, not application tips

---

**u/grupiotr** [OP] (5 points):

John Schulman's talk wins, particularly:

- rescaling observations, rewards, targets and prediction targets
- using big replay buffers, bigger batch size and generally more iterations to start with
- always starting with a simple version of the task to get signs of life

---

**u/Kaixhin** (2 points):

My first bit of advice is actually don't do RL. If the answer is still yes, find some other useful task for the network to do, like predicting something. Get supervised gradients flowing through your network. Training end-to-end on purely an RL signal is impressive, but adding easier learning signals can potentially help a lot.

---

**u/grupiotr** [OP] (1 point):

What turned out to be the game-changer (made my RL agents actually learn something) was **rescaling the reward from [-1, 1] to [0, 1]**. Thanks again to everyone that contributed!
