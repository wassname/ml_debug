Source: http://joschu.net/docs/nuts-and-bolts.pdf
Title: Nuts and Bolts of Deep RL Research - John Schulman (2016)
Fetched-via: bash -c 'uvx "markitdown[pdf]" http://joschu.net/docs/nuts-and-bolts.pdf'
Fetch-status: verbatim

| The Nuts | and Bolts | of Deep   | RL Research |
| -------- | --------- | --------- | ----------- |
|          | John      | Schulman  |             |
|          | December  | 9th, 2016 |             |

Outline
| Approaching           | New Problems |            |
| --------------------- | ------------ | ---------- |
| Ongoing Development   |              | and Tuning |
| General Tuning        | Strategies   | for RL     |
| Policy Gradient       | Strategies   |            |
| Q-Learning Strategies |              |            |
| Miscellaneous         | Advice       |            |

Approaching New Problems

| New Algorithm?             | Use Small | Test Problems |
| -------------------------- | --------- | ------------- |
| (cid:73) Run experiments   | quickly   |               |
| (cid:73) Do hyperparameter | search    |               |
(cid:73) Interpret and visualize learning process: state visitation, value function, etc.
(cid:73) Counterpoint: don’t overfit algorithm to contrived problem
(cid:73) Useful to have medium-sized problems that you’re intimately familiar with
(Hopper, Atari Pong)

| New Task?        | Make            | It Easier Until | Signs | of Life |
| ---------------- | --------------- | --------------- | ----- | ------- |
| (cid:73) Provide | good input      | features        |       |         |
| (cid:73) Shape   | reward function |                 |       |         |

POMDP Design
(cid:73) Visualize random policy: does it sometimes exhibit desired behavior?
| (cid:73) Human | control |     |     |     |
| -------------- | ------- | --- | --- | --- |
(cid:73) Atari: can you see game features in downsampled image?
(cid:73) Plot time series for observations and rewards. Are they on a reasonable
scale?
| (cid:73) hopper.py | in gym:      |                             |         |             |
| ------------------ | ------------ | --------------------------- | ------- | ----------- |
| reward             | = 1.0        | - 1e-3 * np.square(a).sum() | + delta | x / delta t |
| (cid:73) Histogram | observations | and rewards                 |         |             |

Run Your Baselines
| (cid:73) Don’t expect | them to | work with default | parameters |
| --------------------- | ------- | ----------------- | ---------- |
(cid:73) Recommended:
| Cross-entropy | method1 |     |     |
| ------------- | ------- | --- | --- |
(cid:73)
| (cid:73) Well-tuned | policy gradient | method2        |     |
| ------------------- | --------------- | -------------- | --- |
| (cid:73) Well-tuned | Q-learning      | + SARSA method |     |
1Istv´anSzitaandAndr´asL¨orincz(2006).“LearningTetrisusingthenoisycross-entropymethod”. In:Neuralcomputation.
2https://github.com/openai/rllab

| Run with | More Samples | Than | Expected |     |
| -------- | ------------ | ---- | -------- | --- |
(cid:73) Early in tuning process, may need huge number of samples
|     | Don’t be deterred | by published | work |     |
| --- | ----------------- | ------------ | ---- | --- |
(cid:73)
| (cid:73) Examples: |     |     |     |     |
| ------------------ | --- | --- | --- | --- |
(cid:73) TRPO on Atari: 100K timesteps per batch for KL= 0.01
|     | DQN on Atari: | update freq=10K, | replay buffer | size=1M |
| --- | ------------- | ---------------- | ------------- | ------- |
(cid:73)

| Ongoing | Development | and Tuning |
| ------- | ----------- | ---------- |

| It  | Works!           | But         | Don’t | Be Satisfied      |     |     |
| --- | ---------------- | ----------- | ----- | ----------------- | --- | --- |
|     | (cid:73) Explore | sensitivity |       | to each parameter |     |     |
(cid:73) If too sensitive, it doesn’t really work, you just got lucky
|     | (cid:73) Look | for health      | indicators |     |     |     |
| --- | ------------- | --------------- | ---------- | --- | --- | --- |
|     |               | (cid:73) VF fit | quality    |     |     |     |
|     |               | Policy          | entropy    |     |     |     |
(cid:73)
|     |     | (cid:73) Update   | size in     | output space | and parameter | space |
| --- | --- | ----------------- | ----------- | ------------ | ------------- | ----- |
|     |     | (cid:73) Standard | diagnostics | for          | deep networks |       |

| Continually         | Benchmark |               | Your Code    |
| ------------------- | --------- | ------------- | ------------ |
| (cid:73) If reusing | code,     | regressions   | occur        |
| (cid:73) Run        | a battery | of benchmarks | occasionally |

| Always | Use Multiple | Random | Seeds |
| ------ | ------------ | ------ | ----- |

| Always Be          | Ablating   |            |
| ------------------ | ---------- | ---------- |
| (cid:73) Different | tricks may | substitute |
| Especially         | whitening  |            |
(cid:73)
(cid:73) “Regularize” to favor simplicity in algorithm design space
| (cid:73) As | usual, simplicity | → generalization |
| ----------- | ----------------- | ---------------- |

| Automate Your | Experiments      |           |                   |
| ------------- | ---------------- | --------- | ----------------- |
| Don’t spend   | all day watching | your code | print out numbers |
(cid:73)
(cid:73) Consider using a cloud computing platform (Microsoft Azure, Amazon EC2,
| Google Compute | Engine) |     |     |
| -------------- | ------- | --- | --- |

| General | Tuning | Strategies | for RL |
| ------- | ------ | ---------- | ------ |

| Whitening                | / Standardizing | Data               |
| ------------------------ | --------------- | ------------------ |
| (cid:73) If observations | have unknown    | range, standardize |
(cid:73) Compute running estimate of mean and standard deviation
x(cid:48)
(cid:73) = clip((x −µ)/σ,−10,10)
(cid:73) Rescale the rewards, but don’t shift mean, as that affects agent’s will to live
(cid:73) Standardize prediction targets (e.g., value functions) the same way

| Generally | Important       | Parameters    |      |         |           |
| --------- | --------------- | ------------- | ---- | ------- | --------- |
| (cid:73)  | Discount        |               |      |         |           |
|           | (cid:73) Return | = r +γr       | +γ2r | +...    |           |
|           |                 | t t           | t+1  | t+2     |           |
|           | Effective       | time horizon: | 1+γ  | +γ2+··· | = 1/(1−γ) |
(cid:73)
(cid:73) I.e., γ =0.99⇒ ignore rewards delayed by more than 100 timesteps
|     | Low | γ works well | for well-shaped | reward |     |
| --- | --- | ------------ | --------------- | ------ | --- |
(cid:73)
(cid:73) In TD(λ) methods, can get away with high γ when λ < 1
| (cid:73) | Action frequency |            |         |               |     |
| -------- | ---------------- | ---------- | ------- | ------------- | --- |
|          | Solvable         | with human | control | (if possible) |     |
(cid:73)
|     | (cid:73) View | random exploration |     |     |     |
| --- | ------------- | ------------------ | --- | --- | --- |

General RL Diagnostics
(cid:73) Look at min/max/stdev of episode returns, along with mean
(cid:73) Look at episode lengths: sometimes provides additional information
| (cid:73) Solving problem | faster, losing | game slower |
| ------------------------ | -------------- | ----------- |

Policy Gradient Strategies

| Entropy as         | Diagnostic       |         |               |
| ------------------ | ---------------- | ------- | ------------- |
| (cid:73) Premature | drop in policy   | entropy | ⇒ no learning |
| (cid:73) Alleviate | by using entropy | bonus   | or KL penalty |

KL as Diagnostic
(cid:2) (cid:3)
| (cid:73) Compute | KL π | (·|s),π(·|s) |     |
| ---------------- | ---- | ------------ | --- |
old
| (cid:73) KL spike    | ⇒ drastic | loss of performance |               |
| -------------------- | --------- | ------------------- | ------------- |
| (cid:73) No learning | progress  | might mean steps    | are too large |
(cid:73) batchsize=100K converges to different result than batchsize=20K.

| Baseline | Explained | Variance |
| -------- | --------- | -------- |
1−Var[empiricalreturn−predictedvalue]
| (cid:73) | explained variance | =   |
| -------- | ------------------ | --- |
Var[empiricalreturn]

Policy Initialization
(cid:73) More important than in supervised learning: determines initial state
visitation
| (cid:73) Zero | or tiny final layer, | to maximize | entropy |
| ------------- | -------------------- | ----------- | ------- |

| Q-Learning Strategies |     |     |
| --------------------- | --- | --- |
(cid:73) Optimize memory usage carefully: you’ll need it for replay buffer
| (cid:73) Learning    | rate schedules |        |
| -------------------- | -------------- | ------ |
| (cid:73) Exploration | schedules      |        |
| (cid:73) Be patient. | DQN converges  | slowly |
(cid:73) On Atari, often 10-40M frames to get policy much better than random
ThankstoSzymonSidorforsuggestions

Miscellaneous Advice
(cid:73) Read older textbooks and theses, not just conference papers
(cid:73) Don’t get stuck on problems—can’t solve everything at once
| (cid:73) Exploration | problems          | like cart-pole swing-up |
| -------------------- | ----------------- | ----------------------- |
| (cid:73) DQN on      | Atari vs CartPole |                         |

Thanks!
