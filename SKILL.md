---
name: ml-debug
description: "Wassname's practical folklore for debugging ML systems: convergence issues, loss surface analysis, gradient analysis, sweep methodology, and same-seed comparisons. Use when stuck on training, designing sweeps, or analyzing experiment results."
---

# ML Debugging Folklore

Practitioner knowledge that's hard to find in papers. Distilled from Schulman's "Nuts and Bolts" talk, Andy Jones' debugging guide, r/reinforcementlearning threads, competition write-ups, and personal experience. Most multi-source claims are traced to sourced quotes in [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown) (vargdown format); uncovered claims are listed in the [process log](docs/ml_debug_folklore_log.md).

**Caveat:** Most sources are from 2017-2021, predating RLHF, large-scale pretraining, and JAX/PyTorch 2.0 workflows. Core debugging principles (isolation testing, logging, seed variance) are architecture-agnostic and likely durable. Specific RL HP defaults and reward-scaling advice may need updating for modern settings.

**LLM pretraining gap:** For modern transformer pretraining debugging, see [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) (2019; general training workflow, activation/gradient health checks) and [nanochat deepwiki](https://deepwiki.com/karpathy/nanochat) (2026; documents 320+ empirical HP sweeps for training a GPT-2-scale model from scratch, covering MFU monitoring, precision management, BOS-aligned dataloaders, and cross-scale ablation discipline). Evidence files: [karpathy_recipe_training_nn_2019.md](docs/evidence/karpathy_recipe_training_nn_2019.md), [nanochat_deepwiki_llm_pretraining_2026.md](docs/evidence/nanochat_deepwiki_llm_pretraining_2026.md).

The core problem in RL (and to a lesser extent supervised ML): errors aren't local [Jones 2021]. In RL, information flows in a loop (actor -> learner -> actor), so a numerical bug in one spot gets smeared through the whole system in seconds. From outside, everything goes weird at once -- loss explodes, KL collapses, rewards oscillate. You can tell something's wrong but not *what* or *where*.

**When debugging, work in this order:**
1. Run static analysis (grep for silent bugs) -- Part 6.1
2. Run diagnostics (data check, init loss, overfit-one-batch) -- Part 6.2
3. Follow the triage decision tree -- Part 6.3
4. Use mental models to brainstorm hypotheses -- Part 7
5. Only then read Parts 1-5 for deeper understanding of specific issues

---

## Part 1: General ML Debugging

### The hierarchy (work in order, don't skip to hyperparameters)

**Step 1: Verify components in isolation.** [Goodfellow Ch11, CS229]
Most bugs are "doing the wrong calculation." Test each piece independently.

- Network forward pass: feed known inputs, check output shapes and ranges. `assert` shapes everywhere -- `(None,)` vs `(None, 1)` silently broadcasts into `(None, None)`.
- Loss computation: hand-compute a few targets and compare to code output.
- Data pipeline: sample a batch, print it, eyeball it. Are labels aligned with inputs? Are transforms applied correctly?
- Preprocessing: look at your processed inputs as a human. Can *you* solve the task from them? If you downsampled images, can you still tell what's going on?

**Five most common deep learning bugs** [FSDL]: (1) incorrect tensor shapes that fail silently via broadcasting, (2) preprocessing inputs incorrectly (wrong normalization, over-augmentation), (3) incorrect loss function or wrong sign in loss/gradient, (4) forgot to set up train vs eval mode (dropout/batchnorm behave differently), (5) numerical instability (NaN from log(0), overflow, vanishing grads).

**Step 2: Get signs of life on a toy problem. Work the baseline ladder.** [CS231n, FSDL, Goodfellow Ch11]
Before your real task, solve something trivial with the same codebase. This establishes what "healthy" looks like. Run on CartPole (or equivalent) and log the same curves so you know what healthy learning looks like for your setup [reddit]. If it works on the toy but not your real task, the gap is usually scale/normalization, not fundamental correctness.

Also try to overfit to train. If you can't do that, you likely won't be able to generalise. [CS231n: "Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset."] Start with a lightweight implementation (<200 lines of new code), no complicated data pipelines [FSDL]. Build those later once the core works.

**Baseline ladder** (for physics/simulation models -- make each step beat the previous one):
1. Persistence: y(t) = y(t-1). Bar for "does the model capture any dynamics at all?"
2. Exponential decay to steady state (first-order response fit).
3. Linear state-space / OLS on finite differences.
4. Pure data MLP (same architecture, no physics). If PINN doesn't beat this, the physics constraint is hurting.
5. Classical solver with fixed parameters (scipy solve_bvp, ODE, etc.).
6. Classical solver with fitted parameters.
7. Then and only then: PINN / learned physics.

Make complexity pay rent. Every added component (more physics, more dimensions, more losses) should improve a metric you care about. If it doesn't, remove it.

**Step 3: Log everything, look for specific pathologies.** [Goodfellow Ch11, Rahtz 2018, CS231n]

What to log:
- Losses (train and val, per-component if multi-objective)
- Gradient norms (per module if possible)
- Learning rates
- Parameter norms / update magnitudes
- Update-to-data ratio per layer: `((lr * p.grad).std() / p.data.std()).log10()` -- target ~-3 [Karpathy nn-zero-to-hero Lec 4]
- Activation statistics (mean, std, fraction of dead ReLUs, saturation % for tanh)
- Data statistics (input distributions, label distributions)

**Sanity check at init** [CS231n]: verify you get the expected loss at chance performance before training starts. E.g., for 10-class softmax the initial loss should be -ln(0.1) = 2.302 with small random weights. If not, something is wrong with initialization or the loss function. Then verify that increasing regularization increases the loss.

| Symptom | Likely cause |
|---|---|
| Loss stuck from the start | LR too low, bad init, data pipeline broken, wrong loss function |
| Loss decreases then explodes | LR too high, numerical instability (log(0), div by 0), gradient accumulation bug |
| Loss NaN | log(0), 0/0, overflow. Use `log(x.clamp(min=1e-8))`, `1/(std + 1e-5)` |
| Train loss good, val loss bad | Overfitting. More data, regularization, smaller model |
| Loss oscillates wildly | LR too high, batch size too small, data shuffling broken |
| Gradients vanish | Too-deep network without skip connections, saturating activations (tanh with large inputs), bad init |
| Gradients explode | No gradient clipping, learning rate too high, recurrent networks without gradient clipping |
| Different results per seed | Normal if small variance; suspicious if large. Check init sensitivity, batch ordering, floating point nondeterminism |
| Model outputs constant | Dead neurons, vanishing gradients, mode collapse, all-zero init |
| Physics loss low but BCs violated | Gradient imbalance -- PDE residual dominates BC gradient; use adaptive loss weighting or hard BCs |
| PINN worse than pure-data MLP | Wrong equations, bad scaling (forgot to nondimensionalize), or physics constraint fighting the data |
| PINN fails on hard PDE regime, works on easy | Curriculum regularization: start with easy parameters, warm-start and increase to target |
| Scalar parameter (U, alpha) stuck at 0 or bound | Degenerate solution; bound and initialize it, or estimate separately before joint training |

**Step 4: Numerical hygiene.** [CS231n]

```python
# Clamp log values
log_prob = prob.clamp(min=1e-8).log()

# Never divide by zero
ratio = x / (std + 1e-5)

# Clip gradients and LOG the pre-clip norm
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
logger.log("grad_norm", grad_norm)

# Catch NaNs early
assert torch.isfinite(loss), f"Loss is {loss}"

# Verify custom gradients (use float64! relative error plummets from 1e-2 to 1e-8)
torch.autograd.gradcheck(my_custom_fn, inputs.double().requires_grad_(True))
```

Gradient clipping *masks* problems -- always log the pre-clip norm to see if it's constantly being triggered. [CS231n: "the ratio of the update magnitudes to the value magnitudes... should be somewhere around 1e-3."]

**Gradient check thresholds** [CS231n]: use relative error, not absolute. Compare analytic vs numerical gradient using centered difference formula. Relative error > 1e-2 = probably wrong. 1e-4 = uncomfortable. 1e-7 = happy. Before checking: (a) turn off regularization and check data loss alone first (regularization can mask data loss bugs), (b) disable dropout and augmentation, (c) use float64 not float32.

**Step 5: Normalization and Nondimensionalization.** [Schulman 2017, CS231n, FSDL, Slavv 2017]
Most ML training issues trace back to scale problems.

- Input normalization: mean 0, std 1 per feature. Use running statistics over ALL data seen so far, not just recent data [Schulman 2017]. Using only recent data silently changes the input distribution in a way the policy doesn't know about, which can collapse performance. [Schulman slides: "Compute running estimate of mean and standard deviation, x' = clip((x-mu)/sigma, -10, 10)"]
- Schulman: "plot histograms of all observations and rewards and make sure each component has the right mean and standard deviation and doesn't have crazy outliers."
- Layer normalization helps stability.
- For targets/labels: think about whether the scale is reasonable for your loss function.
- **For physics/PDE models (PINNs)**: nondimensionalize *before* training. Raw SI units (Kelvin, Joules, meters) create loss terms with wildly different magnitudes -- this is the multi-scale problem that adaptive weighting tries to fix downstream. Nondimensionalizing fixes it at the source by making all PDE coefficients O(1). Recipe: pick characteristic scales (T_ref, L_ref, etc.), define dimensionless variables (T* = T/T_ref, z* = z/L), substitute into the PDE. The resulting groups (NTU, Biot, etc.) are all O(1).
- **Train/test split**: use temporal split (not random) for time-series or plant data. Random splitting leaks temporal correlation and gives optimistic test RMSE. Conventional: first 75% train, last 25% test.

**Step 6: Check your assumptions about the optimizer.**

- Adam's moment estimates can mask gradient problems. If step statistics look weird, check raw gradients separately.
- `abs_max(param_update)` should be small (e.g., ~1e-3 at LR 1e-2); `mean_square(param_update)` should be very small but substantially smaller than abs_max.
- Supervised learning tricks (batch norm, dropout, big networks) often *don't* transfer to RL. People tried them. They usually don't help.

### Assume you have a bug [Jones 2021, Goodfellow Ch11]

> When their RL implementation doesn't work, people are often keen to either (a) adjust their network architecture or (b) adjust their hyperparameters. On the other hand, they're reluctant to say they've got a bug. Most often, it turns out they've got a bug. -- Andy Jones

Bugs are faster to find and fix than validating that a new architecture is an improvement. Dramatically raise your threshold for "OK, I think this is correct." Neural net components can adapt to compensate for bugs, masking them [Goodfellow Ch11: "If one part is broken, the other parts can adapt and still achieve roughly acceptable performance."]

### Loss curves are a red herring [Jones 2021]

They give global information about performance but don't localize errors. Don't debug by staring at loss curves. Use them *after* you've exhausted better methods. Their main value: splitting performance into "how fast it learns" vs "where it plateaus." [Jones: "The shape of your loss curve says very little about where in your code you've messed up."]

### Pursue anomalies [Jones 2021, Rahtz 2018]

> If you ever see a plot or a behaviour that just *seems weird*, chase right after it. Do not just 'hope it goes away'. -- Andy Jones

That cool new feature you were going to add today? It won't magically fix the anomaly. Give up on your plan and chase the anomaly instead. Rahtz independently calls this "noticing confusion" -- following confusion about a frame-differencing improvement led to finding a normalization bug that had hidden for months.

### With long feedback loops, think more, experiment less [Rahtz 2018]

> Switching from experimenting a lot and thinking a little to experimenting a little and thinking a lot was a key turnaround in productivity. -- Rahtz

When runs take hours, pour time into hypothesis-forming *before* launching. Spend 30-60 minutes mapping out possibilities, ranking them by likelihood given all evidence so far. Reserve experiments for distinguishing between your top hypotheses.

Keep a structured work log for long debugging sessions:
1. What specific output am I working on right now?
2. Thinking out loud -- hypotheses about the current problem
3. Record of currently running experiments with what each one is supposed to answer
4. Results of runs (graphs, observations), separated by type

---

## Part 2: RL-Specific Debugging

> See [rl/SKILL.md](rl/SKILL.md) for the full RL debugging sub-skill: probe environments, reward engineering, diagnostics tables, hyperparameter defaults, and reference implementations.

---

## Sources

**Evidence map**: [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown) traces each claim to verbatim quotes across 21 evidence files in [docs/evidence/](docs/evidence/). Process log at [docs/ml_debug_folklore_log.md](docs/ml_debug_folklore_log.md).

### Debugging deep networks (general)
- Goodfellow et al., Deep Learning Book, "Practical Methodology" chapter: https://www.deeplearningbook.org/
- Stanford CS231n, Neural Networks Part 3: https://cs231n.github.io/neural-networks-3/
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010)
- Josh Tobin, FSDL Spring 2021 Lecture 7 "Troubleshooting Deep Neural Networks": https://fullstackdeeplearning.com/
- Andrew Ng, CS229 Machine Learning Advice: Stanford CS229
- McCandlish & Kaplan, "An Empirical Model of Large-Batch Training" (2018): https://arxiv.org/abs/1812.06162
- Slav Ivanov, "37 Reasons why your Neural Network is not working" (2017): https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

### Tools
- PyTorch memory profiling: https://github.com/Stonesjtu/pytorch_memlab
- GPU profiling: nsight, snakeviz, tuna
- Gradient debugging: `torch.autograd.gradcheck`, `torch.autograd.detect_anomaly()`

---

## Part 3: Loss Surface & Gradient Analysis (No Model Required)

When a loss isn't behaving as expected, don't guess -- visualize the loss surface and check gradient flow directly. This technique uses *synthetic tensors* fed into loss sub-components. No model, no forward pass, no GPU. Pure math.

### The method

1. Identify each loss sub-component as a function of its immediate inputs.
2. Pick 1-2 axes that matter (the "natural axes" you think about when reasoning about the loss).
3. Grid over those axes, feed through the loss, call `.backward()`, collect gradients.
4. Plot: contour heatmap + quiver overlay (negative gradient = optimization direction).
5. Build a summary table: component x representative_input -> loss_value, grad_value. Flag zero or non-finite gradients.

### Pseudocode

```py
# ── 2D loss surface with gradient quiver ──────
def analyze_component(loss_fn, x_range, y_range, n=80):
    xs = torch.linspace(*x_range, n)
    ys = torch.linspace(*y_range, n)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    x_flat = X.flatten().requires_grad_(True)
    y_flat = Y.flatten().requires_grad_(True)

    losses = loss_fn(x_flat, y_flat)       # vectorized, returns (n*n,)
    losses.sum().backward()

    loss_grid = losses.detach().reshape(n, n)
    gx = x_flat.grad.reshape(n, n)
    gy = y_flat.grad.reshape(n, n)

    # contourf(X, Y, loss_grid) + quiver(X, Y, -gx, -gy)
    # negative gradient = direction optimizer moves

# ── Gradient flow verification table ──────────
#
# For each component, evaluate at representative inputs
# (zero, small, converged, degenerate). Report loss + grad.
# Flag: zero grad (dead zone), non-finite (numerical issue).
#
# | Component      | Param    | Input        | Loss     | Grad     |
# |----------------|----------|--------------|----------|----------|
# | barrier_penalty | v       | v=0.0        | +0.000   | +0.000   |  <-- zero grad!
# | barrier_penalty | v       | v=0.5        | +12.50   | +50.00   |
# | pair_loss       | dot_pos | (0.3, -0.3)  | -2.340   | -3.000   |
# | pair_loss       | dot_neg | (0.3, -0.3)  | -2.340   | +3.000   |  <-- antisym, good
# | pair_loss       | dot_pos | (0.0, 0.0)   | +0.000   | +0.000   |  <-- dead at init!
```

### What to look for

| Pattern | Meaning | Action |
|---------|---------|--------|
| Gradient arrows point toward desired region | Loss is well-shaped | Ship it |
| Large flat region (zero gradient) | Dead zone -- optimizer stuck if it lands here | Add curvature, change init, or use different parameterization |
| Gradient magnitude 1000x in one axis vs another | Imbalanced -- one axis dominates | Rescale, use log-space, or normalize |
| Saddle point at origin | Common with product-form losses (A*B) | Switch to additive (log A + log B) for independent gradients |
| Arrows point away from desired region | Loss is wrong or has unexpected local min | Rethink the formula |
| Non-finite values in a region | Numerical issue (log(0), 0/0) | Add eps, clamp, or use log1p |

### The log-space decomposition trick

When your loss involves a product of factors A*B and one factor can be near zero:

```
# BAD: symlog(A * B) -- when B~0, chain rule gives 0 grad to A too
# GOOD: sign * (log|A| + log|B|) -- independent gradients
#   d/dA = 1/A  regardless of B
#   d/dB = 1/B  regardless of A
```

General principle: **if you want gradient to flow independently through two factors, decompose multiplicatively in log space.**

### Structural ceiling analysis

Sometimes a metric is stuck not because the optimizer fails but because the parameterization can't express a higher value. To diagnose:

```py
# 1. Check: is d(loss)/d(metric) large? If yes, optimizer IS trying.
metric = torch.tensor(0.5, requires_grad=True)
loss = loss_fn(metric)
loss.backward()
print(metric.grad)   # if large (e.g. 350x the other gradients), it's trying

# 2. Check: can the parameter CHANGE the metric?
# Trace the chain: loss -> metric -> intermediate -> parameter
# If d(metric)/d(parameter) ~ 0, the param is structurally unable to move it.
# Example: V-rotation can't change output basis (U is fixed) so r_sub is capped.

# 3. Confirm empirically: set the exponent to 0 (disable the term).
# If metric reaches the SAME value, it's purely structural (not learned).
```

### When to use this

- New loss function: always visualize before training. 5 minutes of plotting saves hours of puzzling over curves.
- Metric stuck at a value: distinguish "optimizer can't" from "parameterization can't" from "competing losses cancel out."
- After changing loss formula: verify gradient flow didn't break, especially at the operating point (not just at init).
- Comparing loss variants: grid the same axes for both, compare arrow fields side by side.

---

## Part 4: Experiment Sweeps & Statistical Analysis

Principled hyperparameter sweeps with same-seed comparisons, within-group z-scores, and t-stat stability. This is the difference between "I tried it and it seemed better" and "I have evidence it's reliably better."

### Sweep design (justfile pattern)

Each sweep is a justfile recipe. Key conventions:

```just
set shell := ["bash", "-c"]
SEEDS_4 := "2024 4096 8192 9000"
BASE := "uv run python train.py gemma1b"

# Q: Does rotation type matter? block vs full vs givens.
# Hypothesis: block should balance expressiveness vs cost.
# 12 runs, ~3 hours
sweep-rotation-type:
    #!/usr/bin/env bash
    set -x
    export WANDB_RUN_GROUP="sweep-rotation-type-$(date +%Y%m%d-%H%M)"
    for seed in {{ SEEDS_4 }}; do
        {{ BASE }} --seed=$seed --svd_rotation_type=block
        {{ BASE }} --seed=$seed --svd_rotation_type=full
        {{ BASE }} --seed=$seed --svd_rotation_type=givens
    done
```

**Rules:**
- One WANDB_RUN_GROUP per sweep, timestamped.
- Same seeds across all values within a sweep (enables paired comparison).
- Vary ONE parameter per sweep when possible (all-else-equal). If you must vary two, the analysis script warns about confounders.
- Comment the recipe with: question, hypothesis, run count, time estimate.
- Queue sweeps in a `queue` recipe in priority order.

### Logging to wandb

Every run logs to wandb with: group name, seed, all config as hyperparams, final eval metric (SI = TPR - FPR).

Cache locally as parquet to avoid slow API calls on every analysis:

```py
# download_wandb.py pattern:
#   1. Load cached parquet (if exists)
#   2. Find latest cached run date, subtract safety margin (1 day)
#   3. Fetch only runs newer than that
#   4. Merge (diagonal concat, dedup on run_id)
#   5. Save back to parquet + TSV
#
# Also downloads output.log per run for post-hoc log diagnosis.
```

### Analysis: within-group z-scores -> t-stat

The core insight: don't compare raw SI across groups (different base configs, different dates). Compare *within* each group, then aggregate stability across seeds.

```py
# analyze_results.py pseudocode:

for group in groups:
    for seed in seeds_in_group:
        # 1. Collect all SI values for this (group, seed) combo
        si_values = {param_value: SI for runs matching (group, seed, param)}

        # 2. Compute within-(group,seed) z-score
        mu = mean(si_values)
        sigma = std(si_values)
        z[value] = (si[value] - mu) / sigma
        # This normalizes out seed-level baseline differences

    # 3. Aggregate z-scores across seeds for each param value
    for value in param_values:
        mean_z = mean(z[value] across seeds)
        std_z  = std(z[value] across seeds)
        t_stat = mean_z / (std_z / sqrt(n_seeds))
        # t_stat >> 2: reliably better across seeds
        # t_stat ~ 0: no consistent effect
        # t_stat << -2: reliably worse

    # 4. Also compute linear trend (Pearson r) for numeric params
    #    r > 0: more is better. t_stat on r tests reliability.
```

### Interpreting results

| Metric | What it tells you |
|--------|-------------------|
| SI_mean | Raw effect size (higher = better behavioral control) |
| si_q10, si_q90 | Spread. Wide = seed-sensitive. |
| t_stat | Cross-seed reliability. \|t\| > 2 with 4+ seeds is meaningful. |
| linear r | Monotonic trend. r near +1/-1 with significant t_stat = dose-response. |
| "Also varies" warning | Confounders. Can't attribute effect to this param alone. |

**What you're looking for**: high SI_mean *and* strong t_stat (reliable). A value with SI_mean=20 but t_stat=0.5 is a lucky seed. A value with SI_mean=10 but t_stat=4.0 is a real (if modest) effect.

### Common pitfalls

- **Stale cache**: always `download_wandb.py` before analyzing. Stale cache hides new groups.
- **Cross-group comparisons**: different groups have different base configs. "Group A's best value vs Group B's best value" is apples-to-oranges. Compare within groups.
- **n_seeds=1**: t_stat is NaN. You have one data point. Replicate before concluding.
- **Too many params varied**: if a sweep varies 3 params simultaneously, effects are confounded. Split into separate sweeps.
- **Interpreting NaN SI**: usually means eval crashed or the model diverged. Investigate the run log, don't just skip it.
- **"Fill" sweeps**: if a sweep is 13/16 runs done (missing a seed), run the missing seed in a separate group with a clear name (e.g. `sweep-coh-tau-fill`). The analysis script treats it as a separate group -- you merge mentally.

### The full workflow

```bash
# 1. Design sweep: write justfile recipe with hypothesis
# 2. Run it
just sweep-rotation-type
# 3. Wait for completion, then:
uv run python scripts/download_wandb.py
uv run python scripts/analyze_results.py --after $(date +%Y-%m-%d)
# 4. Read the output:
#    - Group Summary table: SI_mean, n_seeds per group
#    - Param tables: per-value SI with t_stat
#    - Linear trends: dose-response for numeric params
# 5. Record findings in research journal
# 6. Update default config if result is clear and reliable
```

---

## Part 5: Diagnosing "Why Won't This Metric Move?"

A structured decision tree for when a metric is stuck. Applies to any training scenario where a quantity you're optimizing plateaus.

### Step 1: Is the gradient nonzero at the metric level?

```py
metric_val = torch.tensor(current_value, requires_grad=True)
loss = loss_fn(metric_val)
loss.backward()
print(f"d(loss)/d(metric) = {metric_val.grad}")
```

- If ~0: the loss doesn't care about this metric at the current operating point. Likely saturated (log1p of huge value), in a dead zone, or the metric is disconnected from the loss.
- If large: the loss IS trying to move it. Problem is downstream.

### Step 2: Can the parameter change the metric?

Trace the chain rule: `loss -> metric -> ... -> parameter`. The metric is a function of intermediate quantities, which are functions of learned parameters. Check `d(metric)/d(parameter)`:

- Analytically: is there a structural reason this derivative is ~0? (e.g., a rotation of V can't change span(U))
- Empirically: disable the loss term entirely (set coefficient to 0). Does the metric reach the same value? If yes, it's structural -- the optimization never moved it in the first place.

### Step 3: Is something else fighting it?

If gradient is nonzero and the parameter CAN change the metric:
- Check competing loss terms: compute gradient contribution from each loss component separately. If two terms have opposite-sign gradients on the same parameter, they cancel.
- Check optimizer state: AdamW momentum from earlier training may resist direction changes. Try resetting optimizer state or using a warmup schedule.
- Check conditioning: if the metric requires coordinated changes across many parameters (e.g., rotating multiple layers simultaneously), the gradient per-parameter may be too small even though the aggregate signal is large.

### Decision table

| d(loss)/d(metric) | d(metric)/d(param) | Same value without loss term? | Diagnosis |
|---|---|---|---|
| ~0 | any | any | Loss saturated or disconnected. Change loss formula. |
| large | ~0 | yes | Structural ceiling. Change parameterization. |
| large | large | no | Competing losses or optimizer inertia. Isolate. |
| large | large | yes | The term helps but converges to same basin. Coincidence or weak effect. |

---

## Part 6: LLM Debugging Playbook

Concrete procedures for an LLM agent debugging ML code. Work top-to-bottom: static analysis first, then diagnostics, then the decision tree. Don't skip to hyperparameter suggestions.

### 6.1 Static analysis: grep for silent bugs

> See [refs/static_analysis.md](refs/static_analysis.md) for the full list of grep patterns. Categories: shape mismatches, autograd breakers, train/eval mode, in-place ops, double softmax, optimizer step ordering, broadcasting traps, wrong loss sign, frozen params, data leakage, class imbalance.

### 6.2 Diagnostic code snippets

> See [refs/diagnostics.md](refs/diagnostics.md) for copy-paste snippets. Includes: data pipeline sanity check, init loss check (with expected values per loss type), overfit-one-batch test, gradient flow check, NaN/Inf hooks, random input test, prime dimension trick, class imbalance check, confidence-sorted errors, weight/bias distributions.

### 6.3 Triage decision tree

Follow top-to-bottom. Stop at the first match.

```
START
  |
  v
[Exception / traceback?] --yes--> Read the traceback. Fix the error. Done.
  |no
  v
[Loss is NaN/Inf?] --yes--> Attach NaN hooks (6.2). Find first module producing NaN.
  |                          Common: log(0), 0/0, exp(large). Add clamp/eps.
  |no
  v
[Init loss wrong?] --yes--> Check data pipeline (6.2). Check loss function.
  |  (see expected            Check for double softmax (6.1).
  |   values in 6.2)         Check labels match model output format.
  |                          Run random input test (6.2): same loss? -> data destroyed.
  |                          Init loss << expected? -> data leakage (Part 7.4).
  |no
  v
[Can't overfit 1 batch?] --yes--> Run gradient flow check (6.2).
  |                                Any None grads? -> disconnected layer
  |                                All-zero grads? -> dead layer / detach
  |                                Check for autograd breakers (6.1)
  |                                Check optimizer step ordering (6.1)
  |no
  v
[Loss stuck from step 0 (but CAN overfit 1 batch)?] --yes--> LR too low? Try 10x.
  |                                 Frozen params? Check requires_grad (6.1).
  |                                 Wrong loss function?
  |no
  v
[Loss decreases then explodes?] --yes--> LR too high? Try 0.1x.
  |                                       Gradient clipping? Log pre-clip norm.
  |                                       Numerical instability? (log, exp, div)
  |no
  v
[Train loss good, val loss bad?] --yes--> Overfitting. Not a bug.
  |                                        More data, regularization, smaller model.
  |no
  v
[Train loss okay but metric bad?] --yes--> Loss-metric misalignment.
  |                                         Is minimizing the loss equivalent to
  |                                         improving the metric? (Part 5)
  |no
  v
[Model outputs constant?] --yes--> Mode collapse. Check:
  |                                 - Class imbalance? Run label count (6.2).
  |                                 - All-zero init? Run weight check (6.2).
  |                                 - Dead ReLUs (try LeakyReLU)?
  |                                 - Confidence-sorted errors (6.2) reveal pattern?
  |no
  v
[Training is slow but not stuck?] --yes--> Not a bug. Consider:
  |                                          - Batch size (Part 1 Step 6)
  |                                          - Architecture depth/width
  |                                          - Data quality
  |no
  v
[None of the above?]
  Read Part 1 (general) or Part 2 (RL-specific) for deeper diagnostics.
  Log everything (Part 1 Step 3) and pursue anomalies.
```

### 6.4 LLM anti-patterns

Things an LLM should NOT suggest when debugging ML code.

**Don't suggest hyperparameter changes before verifying correctness.**
"Try reducing the learning rate" is the #1 wrong response to any training problem. Verify the code is correct first (Parts 1-2). HP tuning on buggy code wastes time.

**Don't add try/except around training code.**
Training code should crash loudly. A caught exception in a training loop hides the bug and produces silently wrong results. The only exception: checkpoint saving on KeyboardInterrupt.

**Don't suggest "try a different optimizer" as a debugging step.**
If Adam doesn't converge, the problem is almost never the optimizer choice. It's the loss, the data, the architecture, or a bug.

**Don't add .detach() or .item() to "fix" gradient errors.**
If autograd complains, something is wrong with the computation graph. Adding .detach() silences the error by cutting gradient flow -- it doesn't fix anything, it makes the model stop learning from that path. Understand why autograd is complaining first.

**Don't suggest lr_scheduler as a fix for non-convergence.**
Schedulers refine convergence, they don't cause it. If the model doesn't learn with constant LR, a scheduler won't help.

**Don't suggest adding more layers or making the model bigger.**
If the model can't overfit one batch, more parameters won't help. The problem is gradient flow, loss function, or data. Fix those first.

**Don't suggest "normalize your data" without checking if it's already normalized.**
Run the data sanity check (6.2) first. If data is already mean~0, std~1, normalization isn't the problem.

**Don't wrap things in `float()` or `.to(dtype)` to suppress type warnings.**
Type mismatches are signals. A float32/float64 mismatch might mean you're mixing model weights with double-precision data. Fix the root cause.

---

## Part 7: Debugging Folklore & Mental Models

Part 6 tells you what to DO. This part tells you how to THINK. Use these frameworks when generating hypotheses, brainstorming causes, or deciding what to investigate next.

### 7.1 Five mental models for ML debugging

Pick the model that fits your situation. Each gives a different angle on the same problem.

**1. Information flow: trace forward, trace backward.**
Data flows forward through the model; gradients flow backward. A bug anywhere in either direction corrupts everything downstream. When stuck: manually trace shapes and values forward from input through each layer. Then trace gradients backward from loss through each parameter. The break-point is where values go wrong.
- Forward: input -> preprocess -> embed -> layers -> head -> loss
- Backward: loss -> d(loss)/d(head) -> d(head)/d(layers) -> ... -> d(layer1)/d(params)
- Tool: gradient flow check (6.2), NaN hooks (6.2)

**2. Ablation: remove things until it works.** [CS229]
Systematically remove components (regularization, augmentation, auxiliary losses, fancy layers). If removing X fixes the problem, X is the problem. If nothing helps, the bug is in the core (data or main loss).
- Start: turn off ALL regularization, augmentation, dropout, scheduling
- If it works now: add back one-at-a-time until it breaks
- If still broken: problem is in data pipeline, loss, or base architecture
- Tool: just comment things out and rerun overfit-one-batch (6.2)

**3. Oracle substitution: replace each component with ground truth.** [CS229]
For pipeline systems (data -> features -> model -> postprocess -> metric), replace one component at a time with a perfect/oracle version. The component whose oracle gives the biggest accuracy jump is the bottleneck.
- Example: replace learned features with hand-crafted features. Big jump? -> feature learning is the problem.
- Example: replace model predictions with ground truth labels. Small jump? -> model is fine, problem is upstream (data) or downstream (metric).
- This is especially useful for multi-stage systems (NLP pipelines, detection + classification, etc.)

**4. Bias-variance via learning curves.** [CS229, FSDL]
Plot train error and val error as a function of dataset size (or training steps). The shape tells you what to do:
- Both high (converging together): high bias. Model too simple, wrong features, or bug reducing capacity.
- Train low, val high (diverging): high variance. Overfitting. More data, regularization, smaller model.
- Both low: working. Ship it.
- Train low, val high, but val improves with more data: getting there, need more data.
- Val error flat even with 10x more data: not a data problem. Fix the model.

**5. Structural ceiling: can the parameterization express what you want?** (Part 5 expands this)
Sometimes the metric is stuck not because the optimizer fails but because the architecture/parameterization literally cannot represent the desired function. Check: disable the loss term entirely. Does the metric reach the same value? If yes, the loss never moved it -- the model can't express higher values.

### 7.2 Practitioner priors: what's usually wrong

When you have no other information, investigate in this order. Rough estimates synthesized from [Goodfellow, FSDL, Slavv, Jones, CS231n] -- not measured frequencies, just practitioner consensus on what's usually wrong:

1. **Data pipeline** (~40% of bugs). Wrong preprocessing, labels misaligned with inputs, normalization missing or wrong, train/test leakage, data loader returning stale/wrong batches. "It's almost always the data." [FSDL, Slavv]
2. **Loss function** (~20%). Wrong loss for the task, wrong sign, double softmax, loss not connected to metric, competing losses canceling gradients.
3. **Training procedure** (~15%). Wrong optimizer step order, missing zero_grad, wrong LR, frozen params, in-place ops breaking autograd.
4. **Architecture** (~10%). Too small (can't express), too deep (vanishing grads), wrong activation, missing skip connections.
5. **Hyperparameters** (~5%). LR, batch size, weight decay. Almost never the real problem if the code is buggy.
6. **Numerical issues** (~5%). NaN, overflow, underflow. Usually a symptom of something else.
7. **Environment/infrastructure** (~5%). Wrong library version, GPU memory, nondeterminism, stale cache.

For RL specifically, add:
- **Reward scale/sign** as a top-3 issue [Henderson, Schulman]. Rescaling from [-1,1] to [0,1] or vice versa can be the entire difference.
- **Episode boundary handling** (done signals, reward discounting across resets) [Jones].

### 7.3 The debugging mindset

Core attitudes are covered in Part 1 ("Assume you have a bug," "Pursue anomalies," "Loss curves are a red herring") and [Part 2](rl/SKILL.md) ("Working from reference implementations"). Here are the additional mental habits not covered there:

**"Think more, experiment less."** [Rahtz 2018]
When runs take hours, spend 30-60 minutes mapping hypotheses before launching. Rank by likelihood given all evidence. Only run experiments that distinguish between your top hypotheses. Rahtz: "Switching from experimenting a lot and thinking a little to experimenting a little and thinking a lot was a key turnaround."

**MurphyJitsu pre-flight.** [Rahtz 2018]
Before starting a run, ask: "If this run fails, what would the most likely cause be?" If you can name it, test for it first. This is the rationalist habit of "pre-hindsight" -- imagining the failure and working backward.

**"Tricks substitute for each other."** [Schulman 2017]
Many normalization/regularization tricks do roughly the same thing. Adding more tricks adds complexity without proportional benefit. If you have three normalization schemes and the model still doesn't work, the problem isn't normalization.

**Diff against reference implementations.** [Henderson 2018, Jones 2021]
When stuck, diff your code line-by-line against a working reference. The bug is usually in something "trivial" -- episode resets, advantage normalization, dtype. Henderson et al. 2018: "implementation differences which are often not reflected in publications can have dramatic impacts on performance." See [rl/SKILL.md section 9](rl/SKILL.md) for details.

### 7.4 When to suspect the data

Specific signal patterns that point to data problems.

| Signal | Diagnosis | Action |
|--------|-----------|--------|
| Init loss << expected (e.g., 0.01 instead of 2.3) | Data leakage or shortcut. Model "knows" the answer at init. | Check: are labels in the input? Is test data in train? Is there a trivial feature? |
| Random input gives same loss as real input (6.2) | Data pipeline destroying information. Preprocessing too aggressive, wrong transforms, input all zeros. | Print raw data at each pipeline stage. Visualize. |
| Model predicts same class for everything | Class imbalance. 100:1 ratio = model learns "always predict majority." | Run class balance check (6.2). Use weighted loss or resample. |
| Val loss much worse than expected but train is fine | Distribution shift. Val set from different distribution than train. | Check: same preprocessing? Same time period? Same source? Use dual val sets [FSDL]. |
| Learning curve flat even with 10x more data | NOT a data problem. High bias. Model too simple or wrong features. | Add capacity, fix features, check for bugs reducing effective capacity. |
| Adding data makes val WORSE | Data quality issue. New data is noisier or from wrong distribution. | Inspect recent additions. Check label quality. |
| Model works on reference dataset (MNIST/CIFAR) but not yours | Your data is the problem, not the model. | Simplify your data (fewer classes, clean labels, easy examples only). Scale up gradually. [Slavv] |

### 7.5 Batch size & learning rate folklore

These interact in non-obvious ways. Get them wrong and training looks broken even with correct code.

**Critical batch size** [McCandlish 2018]: there's a batch size B_crit below which doubling batch size ~halves training time (compute-efficient), and above which it doesn't help (just wastes compute). B_crit depends on the task and increases during training as the loss decreases.

**LR must scale with batch size.** [McCandlish 2018, Goyal et al. 2017]
- Linear scaling rule (SGD): if you double batch size, double LR. [Goyal et al. 2017]
- For Adam: the scaling exponent is between 0.5 and 1 (between sqrt and linear), task-dependent. [McCandlish 2018]
- Changing batch size without adjusting LR is a common silent mistake.

**Adam default LR = 3e-4.** [FSDL, Karpathy]
This is the "just works" starting point. If you're using Adam and haven't tuned LR, start here. Karpathy: "3e-4 is the best learning rate for Adam."

**Big batches need warmup.** [Goyal et al. 2017]
Large batch training with high LR diverges at the start. Warm up LR linearly over the first few hundred steps. Without warmup, you'll see loss spike/NaN in the first epoch and think the code is broken.

**Batch size signals:**

| Symptom | Likely cause |
|---------|-------------|
| Training very noisy, loss oscillates | Batch too small. Gradient noise overwhelms signal. Try 4-8x larger. |
| Training smooth but slow, poor generalization | Batch too large without LR scaling. Try higher LR or smaller batch. |
| Loss spikes at start then recovers | Normal with large batch + warmup. If no warmup: add it. |
| Different results at different batch sizes (same total steps) | Missing LR scaling. Adjust LR proportionally. |
