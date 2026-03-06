Source: https://old.reddit.com/r/reinforcementlearning/comments/9sh77q/
Title: What are your best tips for debugging RL problems?
Fetched-via: Reddit JSON API (limit=500, depth=10)
Fetch-status: verbatim

# What are your best tips for debugging RL problems?

**Posted by:** u/GrundleMoof | Score: 21 | 8 comments

I've done a few RL toy problems, but I'm still pretty new to the field. In each of the problems I've done, there has been some point where it seems like I've implemented everything correctly, the environment is working correctly, etc, but it's still just not working, or is, with some really strange problem.

RL seems to be harder to debug than any other type of programming I've done before. There's an element of randomness usually. It often takes a while (in the run) for the problem to manifest, so it's hard to pinpoint exactly *where* something is going wrong. Lastly, stuff just takes a while to even run, so my "attempt solution/code/evaluate" loop takes a long time, which makes it even harder.

Does anyone have any tips? The things I've figured out so far are to log everything feasible, and to try to isolate things to find the problem, but those are pretty general tips. I've found some help a few times from reading relevant papers, but that's rarer.

Do any experts in the field have any tips?

## Comments

**u/marcin_gumer** (score: 18):
Hi

This is exactly what I was struggling with for a long time (and still am). RL agent modules are really closely interconnected. No matter which module has issue (neural net, Bellman backups, memory buffer, environment, pre-processing) it will immediately affect all other modules by feeding them bad data. Looking from outside it looks like big gooey mess.

First, I'm not an RL expert, sorry if my advice sounds basic. Couple of things I have learned so far:

* RL is very difficult to debug, especially when neural nets are involved
* DO NOT "try stuff" and run to "see if it works" - this approach doesn't work in RL - too many things need to happen exactly right to see any learning at all
* RL agent modules implementation - this is just good programming practices, but even more important in RL:
   * most modules can be tested independently. Environment, neural net, RL backups, memory reply buffers all can be tested in isolation.
   * I try to unit test everything, usually unit tests take more code than what they test
   * I try to put asserts absolutely everywhere, input matrix dimensions (1d array may broadcast differently than 2d array etc.), input/output ranges (state/actions valid?), output matrix dimensions. Input/output data types (in Python np.ndarray behaves differently than np.matrix in some cases)
* Agent modules integration - generally stepping through code at least once after every change to confirm it is doing what I think it should be doing. It's a bit like programming a bomb detonator or something. Really make sure it is working correctly *before* running long experiment.
* Visualise as much as possible, log absolutely everything
   * record and display agent observations/actions/rewards
      * rewards should have some variance, if all rewards are always equal (e.g. always 0), then there is nothing to learn, it's environment or exploration issue
   * record and display current q-function approximation across whole state space (works only on simple tabular problems and 2d continuous state spaces).
   * pick couple states (can pick by running random policy) and plot predicted q-values over time for these states. q-values should change and stabilise.
   * record inputs/outputs to/from every module (environment, neural net, memory buffer, etc.)
   * neural networks are making everything 10x more difficult - I try to make agent work with linear approximator first (on small problem like mountain car), then when I know everything else is working, swap in neural net and try bigger problem.
   * with neural nets, one can record gradients, individual neuron activations etc to evaluate if neural net is learning over time, even without access to loss function. Debugging neural networks:
      * [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/) "Practical Methodology" chapter
      * [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
      * [https://cs231n.github.io/neural-networks-3](https://cs231n.github.io/neural-networks-3/#baby)/
      * [https://deeplearning4j.org/docs/latest/deeplearning4j-nn-visualization](https://deeplearning4j.org/docs/latest/deeplearning4j-nn-visualization)
* Sometimes It helps to freeze random seed everywhere (numpy, tensorflow, python hashing, gym) and force single threaded CPU execution (to remove randomness from concurrent execution). This way you can reproduce runs exactly to full floating point precision and debug where these NaNs came from or why q-values exploded to infinity etc.
* I would be careful with reference implementations. Some work on hacked environments that are much easier than normal (seen it couple times in blog posts). Or have some "weird" reward function engineering. Or use older version of 3rd party library with easier version of environment. 
* Try hyper parameters from reference implementation.

Hope this helps!

  **u/AlexanderYau** (score: 1):
  Wow, a lot of experience on debugging RL, I can't agree more on "DO NOT "try stuff" and run to "see if it works"".  Have you ever published any paper on RL?

    **u/marcin_gumer** (score: 1):
    I wouldn't say a lot of experience, just figuring things out as I go. I haven't published any RL paper. Currently just building portfolio on my [github.com/marcinbogdanski](https://github.com/marcinbogdanski). But there is not that much there yet, I implemented some algorithms from Sutton &amp; Barto, currently working on DQN and Atari games, but it will take some time.

**u/p-morais** (score: 7):
Here’s a good talk by John Schulman on just this:

https://m.youtube.com/watch?v=8EcdaCk9KaQ

**u/WhichPressure** (score: 5):
I think anyone who touch the RL has the same problem as you! Me too:P This guy wrote nice article about his adventure with some RL side project and he gave some tips. Maybe somehow it'll be helpful for you:  
[http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM](http://amid.fish/reproducing-deep-rl?fbclid=IwAR1VPZm3FSTrV8BZ4UdFc2ExZy0olusmaewmloTPhpA4QOnHKRI2LLOz3mM)

**u/lmericle** (score: 4):
It's true, there's a lot of moving parts. I'm no expert but lately I've been experiencing similar setbacks. 

Typically I find the first place to look is the hyperparameters, and then to consider which particular optimization algorithm you're using and how it might explore the space/optimize toward suboptimal behavior. Next I'd consider the reward function and the interplay between that and the exploration/exploitation behavior of the optimization algorithm. Finally, consider where stochasticity is introduced into the problem -- perhaps there's too much, or not enough, or the stochasticity prevents convergence due to inadequate penalty terms (e.g. low entropy coefficient in PPO).

**u/wassname** (score: 2):
Also checkout [this](https://old.reddit.com/r/reinforcementlearning/comments/7s8px9/deep_reinforcement_learning_practical_tips/) previous discussion.

**u/[deleted]** (score: 1):
I usually just plot every metric, every layer weight on tensorboard and look for anomalies.

&amp;#x200B;
