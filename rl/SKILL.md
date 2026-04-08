---
name: rl
description: "RL-specific debugging: probe environments, reward engineering, diagnostics, hyperparameter defaults, and reference implementations. Sub-skill of ml-debugging. Use when debugging reinforcement learning systems."
---

# RL-Specific Debugging

Everything in the [parent ml_debug skill](../SKILL.md) applies, plus RL has unique challenges:

- **Errors aren't local**: actor -> learner -> actor loop smears bugs everywhere
- **Performance is noisy**: a bug-free implementation can fail due to hyperparameters; a buggy one might seem to work [Irpan 2018: "If my reinforcement learning code does no better than random, I have no idea if it's a bug, if my hyperparameters are bad, or if I simply got unlucky."]
- **Few narrow interfaces**: components consume/produce huge arrays and are heavily stateful
- **Few good abstractions**: you need to understand env, network, optimizer, backprop, multiprocessing, GPU, all at once
- **Compound cost**: sample inefficiency x instability x HP sensitivity multiplies compute needed. 1M steps x 5 seeds x HP tuning = exploding compute to test a single hypothesis [Irpan 2018]

## RL debugging hierarchy

**1. Use probe environments (from Andy Jones).** [Jones 2021]

Don't just test on CartPole -- construct environments that *localize* errors:

1. **One action, zero obs, one step, +1 reward**: Isolates value network. If it can't learn value=1, value loss or optimizer is broken.
2. **One action, random +/-1 obs, one step, obs-dependent +/-1 reward**: If (1) works but not this, backprop through the network is broken.
3. **One action, zero-then-one obs, two steps, +1 reward at end**: If (2) works but not this, reward discounting is broken.
4. **Two actions, zero obs, one step, action-dependent +/-1 reward**: First to exercise policy. If it fails, check advantage calculation, policy loss, or policy update.
5. **Two actions, random +/-1 obs, one step, action-and-obs dependent +/-1 reward**: Policy and value networks interact. Check that policy picks right action per state AND value network learns value=+1.
6. **Progressively harder from there.**

Each env should solve in seconds. If it takes longer, you have a bug.

**2. Use probe agents.**

- **Cheat agents**: Leak extra info (e.g., goal direction). If it can't solve the task even with extra info, the problem is elsewhere.
- **Automatons**: Hand-written algorithms, no NN. Tests that the environment is actually solvable.
- **Tabular agents**: Replace NN with a lookup table on simple envs. Far easier to inspect.

**3. Unit test the tricky bits.** [Jones 2021]

Most bugs cluster in the same few places:
- Reward discounting around episode resets
- Advantage calculations around resets
- Buffering: pairing wrong rewards with wrong observations
- Done signal handling (wrapped envs silently truncate episodes -- if you ignore `done` but store it in your buffer, all updates are wrong)

These are deterministic, easy to factor out, fast to test.

**4. Verify reward and observation scale.**

Schulman: "as a rule of thumb, usually want everything to be mean 0 and standard deviation 1 for observations."

For rewards: hand-scale so that value targets land in [-10, +10], ideally [-3, +3] [Jones 2021]. The hyperparameters from papers are tuned for this range.

> Don't be tempted to write an adaptive reward scaling scheme. It's extra nonstationarity. Just hand-scale. -- Andy Jones

For reward normalization: rescale but DON'T shift the mean [Schulman 2017]. Shifting the mean changes the agent's "will to live" -- how long it wants to survive. You're changing the problem. Henderson et al. 2018 confirm experimentally that reward rescaling can have large effects on DDPG performance.

**5. RL-specific diagnostics to log.** [Schulman 2017, Jones 2021]

| Metric | Healthy behavior | What it tells you |
|---|---|---|
| **Policy entropy** (relative to max) | Starts near 1, falls, flattens | If stays ~1: not learning any policy. If drops to 0: collapsed, not exploring. If oscillates wildly: LR too high. |
| **KL divergence** (old vs new policy) | Small but positive, stable | Very large: stale experience. Very low: can increase LR. Growing over time: feeding same old experience repeatedly. Negative: calculation bug. |
| **Residual variance** (var(target - predicted) / var(target)) | Starts ~1, falls rapidly, then slowly | Stays ~1: value net not learning. Drops to 0: policy collapsed or one scenario dominates. *Negative* explained variance (= 1 - residual_var) means the value net is worse than predicting zero -- likely overfitting or broken [Schulman 2017]. |
| **Value target distribution** | In [-10, +10], ideally [-3, +3] | Blowing up: discount too high or reward discounting broken. |
| **Advantage distribution** | Approximately mean-zero, in [-10, +10] | Persistently non-zero mean: advantage calculation broken. |
| **Episode length** | Depends on env | All episodes same unexpected length: env broken or degenerate policy. |
| **Max episode return** | Look at it, not just mean | If max is high but mean is low, the policy knows the good strategy but can't consistently execute it. Schulman: "if you have a deterministic system, that maximum return is something your policy can hone in on pretty straightforwardly." |
| **Std of policy** (continuous) | Should decrease as it learns (PPO) | Not decreasing: not learning. Collapsed to 0: no exploration. |
| **Critic loss** then **actor loss** | Critic converges first, actor follows | Actor loss initially increases: normal -- value function is a moving target during critic warmup. |
| **Sample staleness** | Steady throughout training | Growing: buffering problem. |

**6. RL hyperparameter defaults (only after verifying correctness).**

| Parameter | Default | Notes |
|---|---|---|
| Hidden layers | 2 x 64 or 2 x 256 | Small networks can accomplish a lot. Jones: 4x256 FC learned *perfect* play on 9x9 board game. |
| Activation | ReLU for hidden (tanh sensitive to init/scale) | tanh on output layers where needed (e.g., action bounds) |
| Optimizer | Adam | |
| gamma (discount) | 0.99 | Think about what 1/(1-gamma) = 100 timesteps means in real time. With TD(lambda), can use gamma->0.999 if lambda<1. |
| Batch size | BIG. Pong ~1k, Space Invaders ~10k, Dota ~100k | Schulman: "sometimes you need to use bigger batch sizes than you thought." TRPO needed 100k. McCandlish et al. 2018 provide theoretical foundation via critical batch size. |
| Critic LR | >= Actor LR | Critic needs to learn first to provide signal |
| Replay buffer | Big as you can afford (DQN: 1M steps) | |
| Entropy coeff | Start 0.01, decrease | |
| Exploration | Epsilon-greedy schedules help for Q-learning | |
| Policy init | Final layer zero or very small | Ensures random exploration initially instead of arbitrary strong opinions |

**7. Reward engineering.** [Irpan 2018, Schulman 2017]

- Rewards must have *variance*. If all rewards are equal, there's nothing to learn.
- Rescaling rewards from [-1, 1] to [0, 1] was the game-changer for at least one practitioner stuck on Pendulum.
- Shaping rewards (e.g., distance to target instead of sparse success/fail) gives much faster learning but changes the problem.
- Don't shift reward mean (changes the MDP).

**Reward misspecification war stories** [Irpan 2018]: (a) Agent trained to navigate a room with no penalty for going out of bounds. Negative reward was plentiful, positive reward too hard. Policy learned to be *suicidal* -- quick death at 0 reward was preferable to a long life risking negative reward. (b) Robot arm reaching toward a point defined relative to a table. Policy learned to slam the table, making it fall over, moving the target point. Both are cases where a minor reward specification error changed the learned optimization target fundamentally.

**8. Environment setup.**

- If all vectorized envs start from the same state, initial batches are highly correlated. Mix envs by taking random steps first. Check: if resets cluster on one timestep, not well-mixed.
- Think about time discretization: can a human control the system at this frame skip rate? What does random exploration look like at this discretization? If you repeat the same action too many times, you get weird Brownian motion.
- Avoid pixels if you can. Before your agent does anything interesting, it has to learn to *see* -- from sparse rewards. Use gridworlds, state vectors, or simple observations first.

**9. Working from reference implementations.** [Jones 2021, Rahtz 2018]

> If you're new to RL, writing things from scratch is the most catastrophically self-sabotaging thing you can do. -- Andy Jones

The allure of writing from scratch is real but the self-correction mechanisms in RL are too weak. Henderson et al. 2018 found that "implementation differences which are often not reflected in publications can have dramatic impacts on performance" -- different implementations of the *same* algorithm with the *same* hyperparameters perform differently. Options, from safest to riskiest:
1. Use reference impl out-of-the-box, make small changes, verify nothing broke
2. Use reference impl as source of reliable components, work to the same API
3. Have one eye on reference while you write your own -- copy hyperparameters, discounting code, termination handling

References: spinning-up (OpenAI), stable-baselines3, cleanrl (single-file per algo), OpenSpiel (multi-agent).

**10. Don't over-interpret noise.** [Schulman 2017, Henderson 2018, Irpan 2018]

Schulman showed 7 MuJoCo tasks x 3 "different algorithms" that were actually the same algorithm with different seeds. Easy to think blue is best on one task, red on another. Need multiple tasks x multiple seeds. Even 20 seeds leaves a pretty big error bar. Henderson et al. 2018 confirmed this with t-tests: "the variance between runs is enough to create statistically different distributions just from varying random seeds." Irpan reports a 30% failure rate across 10 seeds on Pendulum with identical hyperparameters.

Corollary: don't keep adding modifications until your algorithm is complicated. Many tricks substitute for each other (especially normalization tricks). Simplify -- simpler algorithms generalize better. [Schulman slides: "Different tricks may substitute. Especially whitening."]

**11. When you really have no bug.**

Sometimes (rarely) you don't. Schulman:
- Policy gradient: "if it's going to learn it'll learn at the beginning" -- less burn-in than Q-learning
- DQN: has a "serious warmup period" -- the original authors needed patience and bravery
- Some easy problems (cart-pole swing-up) can defeat state-of-the-art algorithms without careful tuning. Don't get stuck on one problem your method fails on.

**12. Meta-advice from Schulman.** [Schulman 2017, CS231n/Bergstra & Bengio 2012]

- HP search: uniform random sampling, look at results, do regression to find which parameters matter, narrow ranges, repeat. "I use the human version of it." CS231n independently recommends: "Prefer random search to grid search" citing Bergstra & Bengio 2012.
- Read older textbooks and theses -- denser source of useful information than conference papers.
- Automate experiments. Don't spend all day watching your code print numbers.
- Have a battery of benchmark problems you run frequently. Easy to overfit one problem.
- Once something works, check sensitivity to every HP. If it's too sensitive, "you probably just got lucky."

---

## Sources

**Evidence map**: [docs/ml_debug_folklore.argdown](../docs/ml_debug_folklore.argdown) traces each claim to verbatim quotes across 21 evidence files in [docs/evidence/](../docs/evidence/). Process log at [docs/ml_debug_folklore_log.md](../docs/ml_debug_folklore_log.md).

### Talks
- Schulman, "Nuts and Bolts of Deep RL Experimentation," Deep RL Bootcamp 2017
  - Video: https://www.youtube.com/watch?v=8EcdaCk9KaQ
  - Summary: https://github.com/williamFalcon/Deep-Reinforcement-Learning-Bootcamp/blob/master/lecture6.md

### Articles
- Andy Jones, "Debugging RL, Without the Agonizing Pain" (2021): https://andyljones.com/posts/rl-debugging.html
- Matthew Rahtz, "Lessons Learned Reproducing a Deep RL Paper" (2018): http://amid.fish/reproducing-deep-rl
- Henderson et al., "Deep Reinforcement Learning that Matters" (2018): https://arxiv.org/abs/1709.06560
- Alex Irpan, "Deep Reinforcement Learning Doesn't Work Yet" (2018): https://www.alexirpan.com/2018/02/14/rl-hard.html
- McCandlish & Kaplan, "An Empirical Model of Large-Batch Training" (2018): https://arxiv.org/abs/1812.06162
- Slav Ivanov, "37 Reasons why your Neural Network is not working" (2017): https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

### Reference implementations
- OpenAI Spinning Up: https://github.com/openai/spinningup
- Stable Baselines3: https://github.com/DLR-RM/stable-baselines3
- CleanRL: https://github.com/vwxyzjn/cleanrl
- OpenSpiel: https://github.com/deepmind/open_spiel

### Reddit threads
- "Deep RL practical tips" (2018): https://old.reddit.com/r/reinforcementlearning/comments/7s8px9/deep_reinforcement_learning_practical_tips/
- "What are your best tips for debugging RL problems?" (2018): https://old.reddit.com/r/reinforcementlearning/comments/9sh77q/what_are_your_best_tips_for_debugging_rl_problems/
- "How to more intelligently debug RL roadblocks?" (2019): https://old.reddit.com/r/reinforcementlearning/comments/bwjp3r/how_to_more_intelligently_debug_rl_roadblocks/

### Other talks/slides
- Schulman, "The Nuts and Bolts of Deep RL Research" (NIPS 2017 Deep RL Workshop): https://www.reddit.com/r/reinforcementlearning/comments/5hereu/the_nuts_and_bolts_of_deep_rl_research_schulman/
- Levine & Finn, ICML 2017 Tutorial: https://www.reddit.com/r/reinforcementlearning/comments/6vcvu1/icml_2017_tutorial_slides_levine_finn_deep/
- Deep RL Bootcamp 2017 (all slides/talks): https://www.reddit.com/r/reinforcementlearning/comments/75m5vd/deep_rl_bootcamp_2017_slides_and_talks/
