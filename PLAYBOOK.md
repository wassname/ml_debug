# ML debugging playbook (long-form reference)

Part of the [ml-debug skill](SKILL.md); the folklore quotes and sources live there. This file holds the synthesized side: mental models, practitioner priors, step catalogs, symptom tables, the agent debugging loop, triage, and anti-patterns. Read these as menus of hypotheses to widen a search, not as authoritative diagnoses; they are distilled from the folklore sources, not quoted from them.

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
| Init loss << expected (e.g. 0.01 vs 2.3) | Leakage or a shortcut: the model "knows" the answer at init | Are labels in the input? Is test data in train? A trivial feature? Localize with the NaN-poisoning tracer or backprop-to-input check ([refs/diagnostics.md](refs/diagnostics.md)) |
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

**Step 2: Get signs of life on a toy problem, and overfit one batch.**[^cs231n][^fsdl] Before the real task, solve something trivial with the same codebase so you know what "healthy" looks like. Then overfit a tiny batch (see the folklore in [SKILL.md](SKILL.md)). Start with a lightweight implementation (<200 lines of new code), no fancy data pipeline; build that later once the core works.

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

**Step 5: Normalization and scale.**[^schulman][^cs231n][^fsdl][^slavv] Most training issues trace back to scale. Normalize inputs to mean 0, std 1 per feature (see Schulman's quote in [SKILL.md](SKILL.md)). For physics/PDE models, nondimensionalize *before* training: raw SI units (Kelvin, Joules, meters) create loss terms with wildly different magnitudes; pick characteristic scales, substitute, and the resulting groups (NTU, Biot) come out O(1). For time-series, use a temporal train/test split, not random, or you leak correlation.

**Step 6: Check your optimizer assumptions.** Adam's moment estimates can mask gradient problems; if step statistics look weird, inspect raw gradients separately. `abs_max(param_update)` should be small (~1e-3 at LR 1e-2). Supervised-learning tricks (batchnorm, dropout, big nets) often *don't* transfer to RL.

---

## When stuck, read a working implementation

After 1-2 diagnostic cycles that don't localize the bug, or whenever you're building something you haven't built before, stop guessing and go read code that already works. Agents tend to skip this for another round of from-scratch guessing, which is usually the worse bet. The folklore is blunt about this: writing RL from scratch is "the most catastrophically self-sabotaging thing you can do," because the self-correction signal is too weak to catch your bugs[^jones].

Use the `gh` skill to find an implementation. Rank candidates by trust signal: community adoption > papers citing it > open source that runs > author reputation > self-reports. A repo other researchers use as a baseline beats a flashy README.

Read it for three things, explicitly:
1. **The algorithm done right.** Diff your math and your computation graph against theirs. The bug is usually something "trivial": a sign, a reset, an off-by-one, an advantage normalization you skipped. Implementation differences that papers never mention dominate results[^henderson].
2. **The engineering tricks the paper omits.** Did they normalize the input? tanh instead of ReLU? mean-pool instead of last-token? only 6 layers? clip to stop gradient saturation? warm-start? an easier dataset than yours? These live in the code, not the abstract, and they're the difference between "works" and "doesn't."
3. **Proven hyperparameters, schedule, and optimizer.** Copy the values known to work before tuning your own. Their LR, warmup, batch size, weight decay, and optimizer are a working starting point you get for free. For LoRA/fine-tuning, the params vary by model, but unsloth and axolotl defaults are good working knowledge: each is backed by a runnable demo notebook, which is a stronger trust signal than any blog post.

For RL specifically, see [rl/SKILL.md](rl/SKILL.md) (spinning-up, stable-baselines3, cleanrl, OpenSpiel).

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
- [refs/diagnostics.md](refs/diagnostics.md) — copy-paste diagnostic snippets (init-loss check, overfit-one-batch, gradient-flow check, NaN hooks, NaN-poisoning leakage tracer, backprop-to-input dependency check, class-imbalance check).
- [rl/SKILL.md](rl/SKILL.md) — RL-specific debugging: probe environments, reward engineering, HP defaults, reference implementations.
- [pinn/SKILL.md](pinn/SKILL.md) — physics-informed-network debugging: nondimensionalization, gradient pathologies, curriculum.

## Links and further reading

Folklore sources (the quotes above trace to these):

[^jones]: Andy Jones, "Debugging RL, Without the Agonizing Pain" — https://andyljones.com/posts/rl-debugging.html ([cache](docs/evidence/andyljones_rl_debugging.md): anomalies L103-109, write-from-scratch L155, assume-bug L176-180, raise-threshold L182, loss-curve L186-188)
[^rahtz]: Matthew Rahtz (Amid Fish), "Lessons Learned Reproducing a Deep RL Paper" — http://amid.fish/reproducing-deep-rl ([cache](docs/evidence/amid_fish_reproducing_deep_rl.md): frame-diff confusion L85-87, investigate-confusion L100-102, think-more L145-153, don't-implement-RL-yourself L497-501)
[^schulman]: John Schulman, "Nuts and Bolts of Deep RL Research" slides — http://joschu.net/docs/nuts-and-bolts.pdf ([cache](docs/evidence/joschu_nuts_and_bolts.md): Always-Be-Ablating L98-101, standardize-observations L118-125; rendered as bullets because the PDF source is slide fragments)
[^henderson]: Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018) — https://arxiv.org/abs/1709.06560 ([cache](docs/evidence/henderson_2018_deep_rl_matters.md): seeds-create-different-distributions L235, implementation-differences L251)
[^cs231n]: Stanford CS231n, "Neural Networks Part 3" — https://cs231n.github.io/neural-networks-3/ ([cache](docs/evidence/cs231n_neural_networks_3.md): overfit-tiny-subset L89)
[^slavv]: Slav Ivanov, "37 Reasons why your Neural Network is not working" (2017) — https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 ([cache](docs/evidence/slavv_37_reasons_nn.md): opening anecdote L19, emergency checklist L45-51)
[^fsdl]: Josh Tobin, Full Stack Deep Learning Spring 2021, Lecture 7 "Troubleshooting DNNs" — https://fullstackdeeplearning.com/spring2021/lecture-7/ ([cache](docs/evidence/fsdl_spring2021_lecture7.md))
[^goodfellow]: Goodfellow, Bengio, Courville, *Deep Learning*, ch. 11 "Practical Methodology" — https://www.deeplearningbook.org/ ([cache](docs/evidence/goodfellow_ch11_practical_methodology.md): one-part-broken-others-adapt L198, weights-adapt-to-compensate L204)
[^cs229]: Andrew Ng, CS229 "Advice for Applying Machine Learning" — https://cs229.stanford.edu/ ([cache](docs/evidence/cs229_ml_advice.md))
[^jaxtyping]: Patrick Kidger, jaxtyping (runtime shape/dtype checking) — https://github.com/patrick-kidger/jaxtyping

For modern transformer pretraining specifically (the sources above predate it), see [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) and the [nanochat deepwiki](https://deepwiki.com/karpathy/nanochat) (320+ empirical HP sweeps for a GPT-2-scale run). Most multi-source claims trace to quotes in [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown) (vargdown); the full evidence set is in [docs/evidence/](docs/evidence/).

Curated by [wassname](https://github.com/wassname). Companion gist: https://gist.github.com/wassname/e45e41f75c0b50e72ec1f4cff811a277
