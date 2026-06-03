---
name: ml-debug
description: "Wassname's practical folklore for debugging ML systems: convergence issues, gradient pathologies, stuck metrics, sweep reliability, and same-seed comparisons. Use when stuck on training, designing sweeps, or analyzing experiment results."
---

# wassname's ML Debugging Folklore

Foreword: In an attempt to upskill the ML debugging on AI coding assistants (and humans), I've collected high quality sources on ML debugging and the mindset and the "taste". When I started ML I went searching for discussions on best practices, and started a few discussions of my own and they helped me a lot, I hope they can help others. This intro is human written, and the below is AI written with human guidance.

## Before you debug: calibrate

The first thing to calibrate is your own behaviour, especially if you're an LLM agent. ML research code is often outside the training distribution: novel losses, custom architectures, methods with no canonical "right answer" you've seen a thousand times. The trained reflex there is to be confident and fast, to pattern-match a symptom to a fix ("loss stuck -> drop the LR") and apply it. Here that reflex works against you. It commits to one hypothesis before you've looked, and a wrong fix on possibly-buggy code wastes a run *and* corrupts your evidence about what's actually happening.

So slow down and widen out. Most of this skill is a set of habits for staying calibrated and keeping your hypothesis space open until the evidence closes it. The habits transfer across timeseries, GANs, OCR, RL, PINNs, puzzles; the specific fixes in the tables below are local to their setting, so treat those tables as a menu of hypotheses to widen your search, not a lookup-and-apply.

## Mental models

How to *think* when generating hypotheses or deciding what to investigate next. Pick the lens that fits; each gives a different angle on the same problem.

**1. Information flow: trace forward, trace backward.** Data flows forward through the model, gradients flow backward. A bug anywhere in either direction corrupts everything downstream. Manually trace shapes and values forward from input through each layer, then trace gradients backward from loss through each parameter. The break-point is where values first go wrong.
- Forward: input -> preprocess -> embed -> layers -> head -> loss
- Backward: loss -> d(loss)/d(head) -> d(head)/d(layers) -> ... -> d(layer1)/d(params)

**2. Ablation: remove things until it works.**[^cs229] Systematically remove components (regularization, augmentation, auxiliary losses, fancy layers). If removing X fixes it, X was the problem. If nothing helps, the bug is in the core (data or main loss). Start by turning off ALL regularization/augmentation/dropout/scheduling; if it works, add back one at a time until it breaks.

**3. Oracle substitution: replace each component with ground truth.**[^cs229] For pipeline systems (data -> features -> model -> postprocess -> metric), swap one component for a perfect version. The component whose oracle gives the biggest jump is the bottleneck. Replace model predictions with ground-truth labels and the metric barely moves? The model's fine; the problem is upstream (data) or downstream (metric).

**4. Bias-variance via learning curves.**[^cs229][^fsdl] Plot train and val error vs dataset size (or steps). Both high and converging together = high bias (too simple, wrong features, or a capacity-reducing bug). Train low, val high = high variance (overfitting). Val flat even with 10x more data = not a data problem, fix the model.

**5. Structural ceiling: can the parameterization even express what you want?** Sometimes a metric is stuck not because the optimizer fails but because the architecture literally cannot represent the target. Quick check: disable the loss term entirely; if the metric reaches the same value, the loss never moved it. Worked example in [refs/metric_stuck.md](refs/metric_stuck.md).

### Practitioner priors: what's usually wrong

With no other information, investigate in this order. Rough consensus from the folklore sources, not measured frequencies, and only a starting weight (a clue that points elsewhere overrides them outright):

1. **Data pipeline** (~40%). Wrong preprocessing, labels misaligned with inputs, missing/wrong normalization, train/test leakage, a loader returning stale batches. It really is usually the data.[^slavv][^fsdl]
2. **Loss function** (~20%). Wrong loss for the task, wrong sign, double softmax, loss disconnected from the metric, competing losses canceling.
3. **Training procedure** (~15%). Wrong optimizer step order, missing `zero_grad`, frozen params, in-place ops breaking autograd.
4. **Architecture** (~10%). Too small to express it, too deep without skips, wrong activation.
5. **Hyperparameters** (~5%). LR, batch size, weight decay. Almost never the real problem if the code is buggy.
6. **Numerical** (~5%). NaN, overflow, underflow, usually a symptom of one of the above.
7. **Environment** (~5%). Library version, GPU memory, nondeterminism, stale cache.

For RL, add reward scale/sign as a top-3 issue, and episode-boundary handling (done signals, discounting across resets).

### When to suspect the data

| Signal | Likely meaning | Check |
|--------|----------------|-------|
| Init loss << expected (e.g. 0.01 vs 2.3) | Leakage or a shortcut: the model "knows" the answer at init | Are labels in the input? Is test data in train? A trivial feature? |
| Random input gives the same loss as real input | Pipeline is destroying information (over-aggressive preprocessing, wrong transforms, all-zero input) | Print raw data at each stage; visualize |
| Predicts the same class for everything | Class imbalance (100:1 -> "always predict majority") | Label-count check; weighted loss or resample |
| Val much worse than train from the start | Distribution shift between splits | Same preprocessing? Same time period? Same source? |
| Learning curve flat even with 10x data | NOT data: high bias | Add capacity, fix features, check for capacity-reducing bugs |
| Adding data makes val worse | Data-quality issue: new data noisier or off-distribution | Inspect recent additions, check label quality |
| Works on MNIST/CIFAR but not your set | Your data is the problem | Simplify your data (fewer classes, clean labels), scale up gradually[^slavv] |

---

## Part 1: General ML debugging

A catalog of small, well-worn checks, in rough dependency order (each assumes the one before). Pull from it; don't run it end-to-end as a ritual.

**Step 1: Verify components in isolation.**[^goodfellow][^cs229] Most bugs are "doing the wrong calculation." Test each piece independently.
- Forward pass: feed known inputs, check output shapes and ranges. `assert` shapes everywhere, since `(None,)` vs `(None, 1)` silently broadcasts into `(None, None)`. (Or make the shapes runtime-checked contracts with jaxtyping[^jaxtyping] + beartype, which turns the #1 silent bug loud.)
- Loss: hand-compute a few targets and compare to code output.
- Data pipeline: sample a batch, print it, eyeball it. Are labels aligned with inputs? Transforms applied correctly?
- Preprocessing: look at processed inputs as a human. Can *you* solve the task from them?

**Five most common deep-learning bugs**[^fsdl]: (1) tensor shapes that fail silently via broadcasting, (2) preprocessing inputs incorrectly (wrong normalization, over-augmentation), (3) wrong loss function or wrong sign, (4) forgetting train vs eval mode (dropout/batchnorm differ), (5) numerical instability (NaN from log(0), overflow, vanishing grads).

**Step 2: Get signs of life on a toy problem, and overfit one batch.**[^cs231n][^fsdl] Before the real task, solve something trivial with the same codebase so you know what "healthy" looks like. Then overfit a tiny batch (see folklore: "overfit one batch first"). Start with a lightweight implementation (<200 lines of new code), no fancy data pipeline; build that later once the core works.

**Baseline ladder** (for physics/simulation models, each step must beat the previous):
1. Persistence: y(t) = y(t-1). Does the model capture *any* dynamics?
2. Exponential decay to steady state (first-order fit).
3. Linear state-space / OLS on finite differences.
4. Pure-data MLP (same architecture, no physics). If a PINN can't beat this, the physics constraint is hurting.
5. Classical solver, fixed parameters (scipy `solve_bvp`, ODE).
6. Classical solver, fitted parameters.
7. Then and only then: PINN / learned physics.

Make complexity pay rent: every added component (physics, dimensions, losses) should improve a metric you care about, or come out.

**Step 3: Log everything, then look for specific pathologies.**[^goodfellow][^rahtz][^cs231n] Log train+val loss (per-component if multi-objective), gradient norms per module, learning rate, parameter-update magnitudes, the update-to-data ratio per layer (`((lr * p.grad).std() / p.data.std()).log10()`, target ~-3), activation stats (mean, std, dead-ReLU fraction, tanh saturation), and input/label distributions.

**Sanity-check the loss at init**[^cs231n]: verify chance-level loss before training. For 10-class softmax the initial loss should be `-ln(0.1) = 2.302` with small random weights. Wrong init loss means a bad initialization or a broken loss. Then check that increasing regularization increases the loss.

| Symptom | Likely cause |
|---|---|
| Loss stuck from the start | LR too low, bad init, data pipeline broken, wrong loss function |
| Loss decreases then explodes | LR too high, numerical instability (log(0), div by 0), gradient-accumulation bug |
| Loss NaN | log(0), 0/0, overflow. Use `log(x.clamp(min=1e-8))`, `1/(std + 1e-5)` |
| Train loss good, val loss bad | Overfitting. More data, regularization, smaller model |
| Loss oscillates wildly | LR too high, batch too small, data shuffling broken |
| Gradients vanish | Too-deep net without skips, saturating activations, bad init |
| Gradients explode | No gradient clipping, LR too high, RNN without clipping |
| Different results per seed | Normal if small; suspicious if large. Check init sensitivity, batch order, nondeterminism |
| Model outputs constant | Dead neurons, vanishing gradients, mode collapse, all-zero init |
| Physics loss low but BCs violated | Gradient imbalance: PDE residual dominates the BC gradient; adaptive weighting or hard BCs |
| PINN worse than pure-data MLP | Wrong equations, bad scaling (forgot to nondimensionalize), or physics fighting the data |
| Scalar parameter stuck at 0 or a bound | Degenerate solution; bound and initialize it, or estimate it separately first |

**Step 4: Numerical hygiene.**[^cs231n]

```python
log_prob  = prob.clamp(min=1e-8).log()                 # clamp log inputs
ratio     = x / (std + 1e-5)                            # never divide by zero
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
logger.log("grad_norm", grad_norm)                      # clip, but LOG the pre-clip norm
assert torch.isfinite(loss), f"Loss is {loss}"          # catch NaNs early
torch.autograd.gradcheck(my_fn, inputs.double().requires_grad_(True))  # float64! 1e-2 -> 1e-8
```

Gradient clipping *masks* problems, so always log the pre-clip norm to see if it fires every step. For a custom gradient, use relative error (centered difference): `>1e-2` probably wrong, `1e-4` uncomfortable, `1e-7` happy; turn off regularization/dropout and use float64 first.

**Step 5: Normalization and scale.**[^schulman][^cs231n][^fsdl][^slavv] Most training issues trace back to scale. Normalize inputs to mean 0, std 1 per feature (see the folklore quote from Schulman on running statistics). For physics/PDE models, nondimensionalize *before* training: raw SI units (Kelvin, Joules, meters) create loss terms with wildly different magnitudes; pick characteristic scales, substitute, and the resulting groups (NTU, Biot) come out O(1). For time-series, use a temporal train/test split, not random, or you leak correlation.

**Step 6: Check your optimizer assumptions.** Adam's moment estimates can mask gradient problems; if step statistics look weird, inspect raw gradients separately. `abs_max(param_update)` should be small (~1e-3 at LR 1e-2). Supervised-learning tricks (batchnorm, dropout, big nets) often *don't* transfer to RL.

---

## When stuck, read a working implementation

After 1-2 diagnostic cycles that don't localize the bug, or whenever you're building something you haven't built before, stop guessing and go read code that already works. Agents tend to skip this for another round of from-scratch guessing, which is usually the worse bet. The folklore is blunt about this: writing RL from scratch is "the most catastrophically self-sabotaging thing you can do," because the self-correction signal is too weak to catch your bugs[^jones].

Use the `gh` skill to find an implementation. Rank candidates by trust signal: community adoption > papers citing it > open source that runs > author reputation > self-reports. A repo other researchers use as a baseline beats a flashy README.

Read it for three things, explicitly:
1. **The algorithm done right.** Diff your math and your computation graph against theirs. The bug is usually something "trivial": a sign, a reset, an off-by-one, an advantage normalization you skipped. Implementation differences that papers never mention dominate results[^henderson].
2. **The engineering tricks the paper omits.** Did they normalize the input? tanh instead of ReLU? mean-pool instead of last-token? only 6 layers? clip to stop gradient saturation? warm-start? an easier dataset than yours? These live in the code, not the abstract, and they're the difference between "works" and "doesn't."
3. **Proven hyperparameters, schedule, and optimizer.** Copy the values known to work before tuning your own. Their LR, warmup, batch size, weight decay, and optimizer are a working starting point you get for free.

For RL specifically, see [rl/SKILL.md](rl/SKILL.md) (spinning-up, stable-baselines3, cleanrl, OpenSpiel).

---

## Folklore

The hard-won lessons, in the words of the people who learned them. Sources and links are collected under [Links](#links-and-further-reading).

### Assume you have a bug

> When their RL implementation doesn't work, people are often keen to either (a) adjust their network architecture or (b) adjust their hyperparameters. On the other hand, they're reluctant to say they've got a bug. Most often, it turns out they've got a bug. Why bugs are so much more common in RL code is discussed above, but there's another advantage to assuming you've got a bug: bugs are a damn sight faster to find and fix than validating that your new architecture is an improvement over the old one.[^jones]

> What I'm advocating for here is not a blind faith in the buginess of your code, but for dramatically raising the threshold at which you start thinking 'OK, I think this is correct.'[^jones]

A bug can also hide, because most ML models have multiple adaptive parts: "If one part is broken, the other parts can adapt and still achieve roughly acceptable performance"[^goodfellow], and it may not show in the output at all. So raise the bar for "correct."

### Never accept the kludge (Patrick Kidger)

Why is research code so reliably buggy? Kidger's blunt answer:

> Academic software is almost always a poorly-maintained kludge of leaky abstractions, awful formatting, and bugs that don't cripple things only because some other bug stops them from doing so.[^kidger]

> This is a systemic professional failing. [...] the overwhelming majority of your time will be spent in front of a screen, staring at code. And yet most of you (yes, you) would not pass muster as a junior developer.[^kidger]

His fix is a posture, "never accept the kludge": messed up your git repo? Find the commands to fix it, "don't just delete it and clone from the remote."[^kidger] The instinct that refuses kludges is the same one that refuses `.detach()`-to-silence-autograd and `except: pass`.

### Broken code fails silently; measure everything (Spinning Up)

Josh Achiam's warning is RL-framed but general:

> broken RL code almost always fails silently, where the code appears to run fine except that the agent never learns how to solve the task.[^spinningup]

So instrument heavily, because "you can't tell it's broken if you can't see that it's breaking,"[^spinningup] and don't trust one passing setup: "sometimes things will work in one environment even when you have a breaking bug, so make sure to test in more than one environment."[^spinningup]

### Loss curves are a red herring

> When someone's RL implementation isn't working, they *luuuuuurv* to copy-paste a screenshot of their loss curve to you. They do this because they know they want a pretty, exponentially-decaying loss curve, and they know what they have *isn't that*. The problem with using the loss curve as an indicator of correctness is somewhat that it's not reliable, but mostly because it doesn't localise errors. The shape of your loss curve says very little about where in your code you've messed up, and so says very little about what you need to change to get things working.[^jones]

Their real value is splitting "how fast it learns" from "where it plateaus." Use them after better methods, not as a first resort.

### Pursue anomalies; investigate confusion

> If you ever see a plot or a behaviour that just *seems weird*, chase right after it! Do not - do *not* - just 'hope it goes away'. Chasing anomalies is one of the most powerful ways to debug your system, because if you've noticed a problem without having had to go look for it, that means it's a *really big problem*. [...] It's really tempting to think that the cool extra functionality you were planning to write today [...] might just magically fix this anomalous behaviour. It won't. Give up on your plan for the day and chase the anomaly instead.[^jones]

> It was only by following that confusion and realising that taking the difference between frames zeroed out the background that gave the hint of a problem with normalization.[^rahtz]
>
> It seems important to really commit yourself to *always* investigate whenever you notice confusion.[^rahtz]

### Think more, experiment less

> Switching from experimenting a lot and thinking a little to experimenting a little and thinking a lot was a key turnaround in productivity. When debugging with long iteration times, you really need to *pour* time into the hypothesis-forming step - thinking about what all the possibilities are, how likely they seem on their own, and how likely they seem in light of everything you've seen so far. Spend as much time as you need, even if it takes 30 minutes, or an hour. Reserve experiments for once you've fleshed out the hypothesis space as thoroughly as possible and know which pieces of evidence would allow you to best distinguish between the different possibilities.[^rahtz]

Corollary, MurphyJitsu pre-flight: before launching a run, ask "if this fails, what's the most likely cause?" If you can name it, test for it first.

### Inspect the data first

> The first step to training a neural net is to not touch any neural net code at all and instead begin by thoroughly inspecting your data. [...] The outliers especially almost always uncover some bugs in data quality or preprocessing.[^karpathy-recipe]

Slavv's "37 reasons" list opens with the same anecdote (gradients flowing, loss falling, predictions all background) and puts "Verify that the input data is correct" and "Start with a really small dataset (2-20 samples). Overfit on it" at the top of its emergency checklist[^slavv]. FSDL names preprocessing and dataset construction as leading silent-failure categories[^fsdl].

### Labels are often wrong (koaning)

Even benchmark data is dirtier than you think. Vincent Warmerdam:

> It turns out that bad labels are a *huge* problem in many popular benchmark datasets.[^koaning]

His cheap way to find them: train a deliberately high-bias model, then sort by where it disagrees with the label while assigning the correct class low confidence (the confidence-sorted-errors trick). The takeaway: "maybe we should spend [...] less time tuning parameters and instead spend it trying to get a more meaningful dataset."[^koaning]

### The tank story: your model learns the confound (gwern)

The canonical data-leakage parable:

> A cautionary tale in artificial intelligence tells about researchers training an neural network (NN) to detect tanks in photographs, succeeding, only to realize the photographs had been collected under specific conditions for tanks/non-tanks and the NN had learned something useless like time of day.[^gwern]

gwern traced versions back to 1992 and concluded it is "a classic 'urban legend'" with no solid source[^gwern]. The lesson holds twice over: a model will gladly learn a confound in how the data was collected instead of the task (dataset bias / leakage), and even your cautionary tales deserve a citation.

### Read what you actually wrote, not what you meant (gwern)

You can't see your own work clearly, which is why fresh eyes (or a fresh-eyes subagent) catch what you can't:

> you can't find typos in your own writing without a great deal of effort because you know what it's *supposed* to say; so copyediting advice runs like 'read it out loud' or 'print it out and read it' or 'wait a week' [...] or even 'read it upside down'. That's the sort of thing it takes to force you to read what you actually wrote, and not what you thought you wrote.[^gwern-unseeing]

### Overfit one batch first

> Overfit a tiny subset of data. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it's also best to set regularization to zero [...]. Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset.[^cs231n]

> Overfit a single batch of only a few examples (e.g. as little as two). [...] If they do not, there is a bug somewhere and we cannot continue to the next stage.[^karpathy-recipe]

And remove a variable while you're at it: "Always use a fixed random seed [...]. This removes a factor of variation and will help keep you sane."[^karpathy-recipe]

### Normalize and scale everything

From the slides[^schulman] (bullet points, de-artifacted from the PDF):
> - If observations have unknown range, standardize
> - Compute running estimate of mean and standard deviation
> - x' = clip((x - mu)/sigma, -10, 10)
> - Rescale the rewards, but don't shift mean, as that affects agent's will to live
> - Standardize prediction targets (e.g., value functions) the same way

Use running statistics over *all* data seen so far, not just recent data; using only recent data silently shifts the input distribution out from under the model.

### Tricks substitute for each other

On the slides[^schulman]:
> Always Be Ablating
> - Different tricks may substitute
> - Especially whitening

Many normalization/regularization tricks do roughly the same job (they improve conditioning), so stacking them adds complexity without proportional benefit. If you have three normalization schemes and it still doesn't work, the problem isn't normalization. So ablate: most of the things you added are probably unnecessary.

### Don't write RL from scratch; diff against a reference

> If you're doing anything that involves an RL algorithm as a component in a larger system, don't try and implement the RL algorithm yourself. [...] RL is unstable enough at the moment that you'll never be sure whether your system doesn't work because of a bug in your RL implementation or because of a bug in your larger system.[^rahtz]

> We find that implementation differences which are often not reflected in publications can have dramatic impacts on performance.[^henderson]

### Seed variance: you can't tell a bug from bad luck

> Look, there's variance in supervised learning too, but it's rarely this bad. If my supervised learning code failed to beat random chance 30% of the time, I'd have super high confidence there was a bug in data loading or training. If my reinforcement learning code does no better than random, I have no idea if it's a bug, if my hyperparameters are bad, or if I simply got unlucky.[^irpan]

> Instability to random seed is like a canary in a coal mine. If pure randomness is enough to lead to this much variance between runs, imagine how much an actual difference in the code could make.[^irpan]

Henderson confirmed it quantitatively: splitting 10 same-config runs (differing only in seed) into two groups of five produces "statistically different distributions just from varying random seeds."[^henderson] This is why one good run proves nothing, and why sweeps need same-seed pairing and a cross-seed reliability test ([refs/sweeps.md](refs/sweeps.md)).

### 3e-4, and learning-rate folklore

The most-quoted line in the genre is Karpathy's tweet, "3e-4 is the best learning rate for Adam, hands down."[^karpathy-3e4] He confirmed in the same thread that it was a joke, but it stuck because it's a decent default. Read it next to what he actually does in the recipe:

> In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate.[^karpathy-recipe]

So: 3e-4 is a fine *starting* LR for Adam, not a law. The real folklore is "Adam is forgiving, so start there and stop fiddling." It has exceptions, and the biggest is batch size:
- There's a critical batch size below which doubling the batch ~halves wall-clock time, and above which it just burns compute; it rises during training as loss falls[^mccandlish].
- LR must scale with batch size: linearly for SGD (double batch, double LR)[^goyal]; for Adam, an exponent between 0.5 and 1, task-dependent[^mccandlish]. Changing batch size without adjusting LR is a common silent mistake.
- Large-batch + high-LR diverges at the start without warmup[^goyal]. No warmup -> first-epoch loss spike/NaN -> you wrongly think the code is broken.

| Symptom | Likely cause |
|---------|-------------|
| Very noisy, loss oscillates | Batch too small; gradient noise swamps signal. Try 4-8x larger |
| Smooth but slow, poor generalization | Batch too large without LR scaling. Higher LR or smaller batch |
| Loss spikes at start then recovers | Normal with large batch + warmup. No warmup? Add it |
| Different results at different batch sizes (same total steps) | Missing LR scaling. Adjust LR proportionally |

### Tricks hide in reference code (lucidrains)

lucidrains' x-transformers is a catalogue of training tricks, each tied to its paper. The debugging-relevant one: when a transformer diverges, attention logits blowing up is a prime suspect, and the now-standard fix is QK normalization (L2-normalize queries and keys before the dot product).

> We are nearing the point of wiping out a source of transformer training instability with one simple intervention.[^lucidrains]

Scaled-up recipes accumulate these one-line stability fixes in code long before they're written up, which is the whole case for reading a working implementation.

### Modern LLM-pretraining gotchas (nanochat)

Karpathy's nanochat is one of the few public records of what scaling a transformer from scratch actually takes. Two gotchas worth stealing:

> The 'lower validation loss' from BOS-alignment is misleading—it's just fewer noisy tokens, not better learning.[^nanochat]

> If any rank's gradient contains inf, all ranks must clip to avoid divergence.[^nanochat]

The first is a fake-metric-improvement trap (a better number that isn't better learning); the second is a multi-GPU bug that single-GPU testing hides.

---

## Research taste (adjacent to debugging)

Debugging taste and research taste are the same muscle: stay skeptical of your own results, and build a real model of your system instead of pattern-matching.

### Default to disbelieving your own results (Neel Nanda)

> The default state of the world is that your research is false, because doing research is hard.[^nanda]

> Excitement is evidence of bullshit: Generally, most true results are not exciting, but a fair amount of false results are. So from a Bayesian perspective, if a result is exciting and cool, it's even more likely to be false than normal![^nanda]

The cheapest antidote he gives: "Read your data ... Often, the quality of the data is a crucial driver of the results of your experiments. Often, it is quite bad."[^nanda]

### Understand the system to shrink the search (Ulisse Mini)

> When good programmers debug hard problems fast, it's usually because they understand the system well enough to *track the important internal state* in their head, letting them drastically *reduce the solution space they're searching over.*[^ulisse]

### Gears beat black boxes (John Wentworth)

> figuring out a system's gears takes extra work up-front, but yields dividends forever. [...] The black-box approach is cheaper for one-off tasks, but usually doesn't yield any insights which will generalize to new tasks using the same system[^wentworth]

The pattern-matched fix is the black box; a mechanistic model of your system is the capital investment that pays off across many bugs.

---

## For LLM agents

Unfortunately, agents need these procedural mindset-shifts spelled out. This is the babysitting layer, not the durable folklore, hence its place at the bottom. If you're an agent debugging ML code, run the loop and avoid the anti-patterns.

### The debugging loop (use judgment, it's not a checklist)

Roughly in this order, though the point is the underlying mindset:

**Collect clues before theorizing.** Read the traceback and logs. Run static analysis ([refs/static_analysis.md](refs/static_analysis.md)) and the cheap diagnostics ([refs/diagnostics.md](refs/diagnostics.md): data sanity check, init-loss check, overfit-one-batch). If you catch yourself proposing a fix before you've looked at anything, stop.

**Hold several hypotheses at once; resist converging early.** Unless the cause is already obvious (a traceback usually points right at it), generate a few genuinely different explanations before ranking any, so you don't marry the first one. Use the five lenses in Mental models. Then sanity-check yourself with the failure-mode triplet (same idiom as the `research-journal` skill):
- *Likely*: your strongest competitor explanation, with a rough credence.
- *Subtle*: the sneaky one, like sample size, leakage, a confound, a metric artifact, or plain seed variance masquerading as signal.
- *Null*: there's no real effect, or it comes from something else you also changed.

Give each a one-line prior and its cheapest falsifier (`Check: ...`). Anchor priors on Practitioner priors above, but a clue that points elsewhere overrides them outright. Keep observations (reproducible, auditable) separate from inferences, so you can rethink without degrading the evidence.

**Run the cheapest observation that splits your top hypotheses.** Not the most thorough experiment, the most *discriminating* one. Forward-predict each hypothesis ("what would I see if this were the cause?"); a test is strong evidence only where the predictions diverge. A grad-norm line reading ~0 under "dead layer" but healthy under "LR too low" beats a 4-hour sweep that only confirms what you believed.

**Bisect the path to localize where it breaks.** Data flows forward and gradients backward in a chain (input -> preprocess -> layers -> loss -> grads), so probe the midpoint: is the value or gradient already wrong halfway through? Each probe halves the search space. Finding the first module to produce a non-finite value is one case; the same bisection works for finite-but-wrong values, exploded norms, and dead activations.

**Then act, only on what the observation pointed to.** If a cycle or two hasn't localized it, stop tuning and go read working code, which usually beats another guess.

```py
# ── ML debugging loop ────────────────────────
def debug(symptom):
    clues ← collect(traceback, logs, static_analysis, cheap_diagnostics)  # look before theorizing
    H     ← generate(clues, lenses=5) | {likely, subtle, null}            # ≥3 genuinely different
    prior ← anchor(H)              # base rates: data .40 loss .20 train .15 arch .10 hp .05

    while not localized:
        t̂     ← argmax(divergence(predict(h, t) for h in H) / cost(t) for t in candidates)
        obs   ← run(t̂)            # one log line or toy run; keep obs apart from inference
        prior ← update(prior, obs)
        H     ← bisect_path(H, obs)  # halve the search space each probe
        if cycles ≥ 2:
            return read_working_code()   # diff your math + graph vs a trusted impl

    fix(root_cause); assert reproduces(obs)   # no silent fallback; crash if it doesn't
```

### Triage (a menu, not a flowchart to obey)

Rough order to consider, not authoritative; it may not fit your project. Stop when a question fits.

1. Exception/traceback? Read it, fix it, done.
2. Loss NaN/Inf? Attach NaN hooks ([refs/diagnostics.md](refs/diagnostics.md)), find the first module producing NaN. Usual causes: log(0), 0/0, exp(large); add clamp/eps.
3. Init loss wrong? Check the data pipeline and loss; check for double softmax; check labels match output format. Same loss on random input -> data destroyed. Init loss << expected -> leakage.
4. Can't overfit one batch? Gradient-flow check: None grads -> disconnected layer; all-zero grads -> dead layer / detach. Check autograd breakers and optimizer step order.
5. Loss stuck from step 0 but you *can* overfit one batch? LR too low (try 10x), frozen params (check `requires_grad`), wrong loss.
6. Loss decreases then explodes? LR too high (try 0.1x), log the pre-clip grad norm, hunt numerical instability.
7. Train good, val bad? Overfitting, not a bug. More data, regularization, smaller model.
8. Train loss fine but the metric is bad? Loss-metric misalignment ([refs/metric_stuck.md](refs/metric_stuck.md)).
9. Outputs constant? Mode collapse: class imbalance, all-zero init, dead ReLUs, look at confidence-sorted errors.
10. Slow but not stuck? Not a bug. Consider batch size, depth/width, data quality.

### Anti-patterns

These are the overconfident reflexes the "calibrate" section warns about, made concrete. Every one changes behaviour before localizing the bug. (As people put it: "this is sklearn slop," or "the LLM is tweaking hyperparameters like it's in a hackathon, not understanding the problem.")

- Hyperparameter changes before verifying correctness. "Try reducing the learning rate" is the #1 wrong response. Verify the code first; HP tuning on buggy code wastes time.
- `try/except` around training code. Training should crash loudly. A caught exception hides the bug and produces silently wrong results. The one exception is checkpoint-on-KeyboardInterrupt.
- "Try a different optimizer." If Adam doesn't converge, it's almost never the optimizer; it's the loss, the data, the architecture, or a bug.
- `.detach()` / `.item()` to "fix" gradient errors. If autograd complains, the graph is wrong. Detaching silences it by cutting gradient flow, so the model just stops learning from that path.
- `lr_scheduler` as a *cure for non-convergence*. Schedules matter (transformers need warmup, cyclic/cosine is often best-in-class, AdamW is the standard pairing), but they refine or enable convergence in an otherwise-healthy setup; they don't rescue a model that can't learn at constant LR because of a bug. Add the schedule once the basics work, not as a debugging band-aid.
- More layers / a bigger model. If it can't overfit one batch, more parameters won't help. The problem is gradient flow, loss, or data.
- "Normalize your data" without checking whether it already is. Run the data sanity check first.
- `float()` / `.to(dtype)` to suppress type warnings. Type mismatches are signals; a float32/float64 mismatch might mean you're mixing model weights with double-precision data. Fix the root cause.

---

## Appendix: deeper tricks

Look these up when the symptom calls for them; they're kept out of the main flow on purpose.

- [refs/loss_surface.md](refs/loss_surface.md) — visualize a loss surface and its gradient field with synthetic tensors, no model or GPU. For when a custom loss misbehaves.
- [refs/metric_stuck.md](refs/metric_stuck.md) — "why won't this metric move?" plus the structural-ceiling check (is the optimizer failing, or can the parameterization not express it?).
- [refs/sweeps.md](refs/sweeps.md) — same-seed paired comparison and cross-seed t-stat reliability, so a result is "reliably better" not "a lucky seed."
- [refs/static_analysis.md](refs/static_analysis.md) — grep patterns for silent bugs (shape mismatches, autograd breakers, double softmax, step ordering, leakage).
- [refs/diagnostics.md](refs/diagnostics.md) — copy-paste diagnostic snippets (init-loss check, overfit-one-batch, gradient-flow check, NaN hooks, class-imbalance check).
- [rl/SKILL.md](rl/SKILL.md) — RL-specific debugging: probe environments, reward engineering, HP defaults, reference implementations.
- [pinn/SKILL.md](pinn/SKILL.md) — physics-informed-network debugging: nondimensionalization, gradient pathologies, curriculum.

## Links and further reading

Folklore sources (the quotes above trace to these):

[^jones]: Andy Jones, "Debugging RL, Without the Agonizing Pain" — https://andyljones.com/posts/rl-debugging.html ([cache](docs/evidence/andyljones_rl_debugging.md): anomalies L103-109, write-from-scratch L155, assume-bug L176-180, raise-threshold L182, loss-curve L186-188)
[^rahtz]: Matthew Rahtz (Amid Fish), "Lessons Learned Reproducing a Deep RL Paper" — http://amid.fish/reproducing-deep-rl ([cache](docs/evidence/amid_fish_reproducing_deep_rl.md): frame-diff confusion L85-87, investigate-confusion L100-102, think-more L145-153, don't-implement-RL-yourself L497-501)
[^karpathy-recipe]: Andrej Karpathy, "A Recipe for Training Neural Networks" (2019) — https://karpathy.github.io/2019/04/25/recipe/ ([cache](docs/evidence/karpathy_recipe_training_nn_2019.md): inspect-data L26+L32, fixed-seed L39, overfit-one-batch L51, Adam-3e-4 L73; note: this is an abridged note with its own "..." elisions)
[^karpathy-3e4]: Andrej Karpathy, tweet, 23 Nov 2016: "3e-4 is the best learning rate for Adam, hands down." — https://x.com/karpathy/status/801621764144971776 (he confirmed in-thread it was a joke; not in the local evidence files, verified against the tweet)
[^schulman]: John Schulman, "Nuts and Bolts of Deep RL Research" slides — http://joschu.net/docs/nuts-and-bolts.pdf ([cache](docs/evidence/joschu_nuts_and_bolts.md): Always-Be-Ablating L98-101, standardize-observations L118-125; rendered as bullets because the PDF source is slide fragments)
[^henderson]: Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018) — https://arxiv.org/abs/1709.06560 ([cache](docs/evidence/henderson_2018_deep_rl_matters.md): seeds-create-different-distributions L235, implementation-differences L251)
[^irpan]: Alex Irpan, "Deep Reinforcement Learning Doesn't Work Yet" (2018) — https://www.alexirpan.com/2018/02/14/rl-hard.html ([cache](docs/evidence/alexirpan_rl_hard.md): variance-bug-or-unlucky L674-678, seed-canary L705-707)
[^cs231n]: Stanford CS231n, "Neural Networks Part 3" — https://cs231n.github.io/neural-networks-3/ ([cache](docs/evidence/cs231n_neural_networks_3.md): overfit-tiny-subset L89)
[^slavv]: Slav Ivanov, "37 Reasons why your Neural Network is not working" (2017) — https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 ([cache](docs/evidence/slavv_37_reasons_nn.md): opening anecdote L19, emergency checklist L45-51)
[^fsdl]: Josh Tobin, Full Stack Deep Learning Spring 2021, Lecture 7 "Troubleshooting DNNs" — https://fullstackdeeplearning.com/spring2021/lecture-7/ ([cache](docs/evidence/fsdl_spring2021_lecture7.md))
[^goodfellow]: Goodfellow, Bengio, Courville, *Deep Learning*, ch. 11 "Practical Methodology" — https://www.deeplearningbook.org/ ([cache](docs/evidence/goodfellow_ch11_practical_methodology.md): one-part-broken-others-adapt L198, weights-adapt-to-compensate L204)
[^cs229]: Andrew Ng, CS229 "Advice for Applying Machine Learning" — https://cs229.stanford.edu/ ([cache](docs/evidence/cs229_ml_advice.md))
[^mccandlish]: McCandlish, Kaplan et al., "An Empirical Model of Large-Batch Training" (2018) — https://arxiv.org/abs/1812.06162 ([cache](docs/evidence/mccandlish_2018_large_batch.md))
[^goyal]: Goyal et al., "Accurate, Large Minibatch SGD" (2017) — https://arxiv.org/abs/1706.02677
[^lucidrains]: Phil Wang (lucidrains), x-transformers README — https://github.com/lucidrains/x-transformers ([cache](docs/evidence/lucidrains_x_transformers_readme.md): post-embedding LayerNorm / BLOOM+YaLM L366, attention-overflow / cosine-sim norm L1230, autoregressive validation L1234, "wiping out a source of instability" / QK RMSNorm L1292)
[^koaning]: Vincent D. Warmerdam (koaning), "Bad Labels" (2021) — https://koaning.io/posts/labels/ ([cache](docs/evidence/koaning_bad_labels.md): bad-labels-huge-problem L13, confidence-sort trick L21, spend-less-time-tuning L33)
[^jaxtyping]: Patrick Kidger, jaxtyping (runtime shape/dtype checking) — https://github.com/patrick-kidger/jaxtyping
[^nanochat]: nanochat (Karpathy), documented via DeepWiki — https://deepwiki.com/karpathy/nanochat ([cache](docs/evidence/nanochat_deepwiki_llm_pretraining_2026.md): BOS fake-improvement L97, all-ranks-clip-on-inf L131)
[^kidger]: Patrick Kidger, "Just Know Stuff" (2023) — https://kidger.site/thoughts/just-know-stuff/ ([cache](docs/evidence/kidger_just_know_stuff.md): kludge-definition L7, junior-developer L9, never-accept-the-kludge L11, don't-delete-and-clone L13)
[^gwern]: Gwern Branwen, "The Neural Net Tank Legend" — https://gwern.net/tank ([cache](docs/evidence/gwern_tank.md): cautionary tale L7, urban-legend conclusion L9)
[^spinningup]: Joshua Achiam, "Spinning Up as a Deep RL Researcher" (OpenAI, 2018) — https://spinningup.openai.com/en/latest/spinningup/spinningup.html ([cache](docs/evidence/spinningup_researcher.md): fails-silently L11, test-more-than-one-env L19, measure-everything L21)
[^nanda]: Neel Nanda, "How to Become a Mechanistic Interpretability Researcher" — https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher ([cache](docs/evidence/nanda_how_to_mech_interp.md): research-is-false L7, excitement-is-bullshit L9, read-your-data L11)
[^gwern-unseeing]: Gwern Branwen, "Unseeing" — https://gwern.net/unseeing ([cache](docs/evidence/gwern_unseeing.md): read-what-you-wrote L9, single-anomaly L13)
[^ulisse]: Ulisse Mini, "How to get good at programming" — https://www.lesswrong.com/posts/LTypqBMTSmRrrhb2v/how-to-get-good-at-programming ([cache](docs/evidence/ulisse_how_to_get_good_at_programming.md): track-internal-state L7, brute-force-search L9, leaky-abstractions L11)
[^wentworth]: John Wentworth, "Gears-Level Models are Capital Investments" — https://www.lesswrong.com/posts/nEBbw2Bc2CnN2RMxy/gears-level-models-are-capital-investments ([cache](docs/evidence/wentworth_gears_level_models.md): gears-dividends L7, valley-of-bad-theory L11)

For modern transformer pretraining specifically (the sources above predate it), see [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) and the [nanochat deepwiki](https://deepwiki.com/karpathy/nanochat) (320+ empirical HP sweeps for a GPT-2-scale run). Most multi-source claims trace to quotes in [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown) (vargdown); the full evidence set is in [docs/evidence/](docs/evidence/).

Curated by [wassname](https://github.com/wassname). Companion gist: https://gist.github.com/wassname/e45e41f75c0b50e72ec1f4cff811a277
