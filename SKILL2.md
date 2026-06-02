---
name: ml-debug
description: "Wassname's practical folklore for debugging ML systems: convergence, gradients, loss surfaces, sweeps, same-seed comparisons. Use when stuck on training, designing sweeps, or analyzing results. Condensed anchor; deep tables and methodology are linked."
---

# ML debugging folklore (anchor)

Condensed from Schulman's "Nuts and Bolts", Andy Jones' debugging guide, Karpathy's recipe, r/reinforcementlearning, competition write-ups, and personal experience. The tables, triage tree, and sweep methodology are one hop away (see Reference). This page is the part that changes how you debug.

## Calibrate first

If you're an LLM agent, start by calibrating yourself. ML research code is often outside your training distribution: novel losses, custom architectures, methods with no canonical right answer you've seen a thousand times. Your trained reflex is to be fast and confident, pattern-matching a symptom to a fix ("loss stuck -> drop the LR") and applying it. On possibly-buggy research code that reflex burns a run and corrupts the evidence you need to find the real cause. Slow down and widen your hypotheses before you touch anything.

Two moves the model skips by default, and they are the highest-leverage ones:

- Assume you have a bug [Jones]. A bug is faster to find than a new architecture is to validate, and healthy components adapt to mask a broken one. Raise your bar for "this is correct".
- When stuck, read a working implementation. After a cycle or two that doesn't localize the bug, stop guessing and diff your math, computation graph, and hyperparameters against code that already runs. Rank candidates by trust signal (adoption > papers citing it > code that runs > author reputation). More in [reading working code](SKILL.md#when-stuck-read-a-working-implementation).

## The loop

```py
# ── ML debugging loop ────────────────────────
def debug(symptom):
    clues ← collect(traceback, logs, static_analysis, cheap_diagnostics)  # look before theorizing
    H     ← generate(clues, lenses=5) | {likely, subtle, null}            # ≥3 genuinely different
    prior ← anchor(H)              # base rates: data .40 loss .20 train .15 arch .10 hp .05

    while not localized:
        # the cheapest test whose outcome the hypotheses disagree on
        t̂     ← argmax(divergence(predict(h, t) for h in H) / cost(t) for t in candidates)
        obs   ← run(t̂)            # one log line or toy run; keep obs apart from inference
        prior ← update(prior, obs)
        H     ← bisect_path(H, obs)  # halve the search space each probe
        if cycles ≥ 2:
            return read_working_code()

    fix(root_cause); assert reproduces(obs)   # no silent fallback; crash if it doesn't
```

In words:

- Collect clues before theorizing. Read the traceback and logs, run [static analysis](refs/static_analysis.md) and the [cheap diagnostics](refs/diagnostics.md): data sanity, init-loss, overfit-one-batch.
- Hold several hypotheses. Generate a few genuinely different explanations, then attach a failure-mode triplet: likely (your strongest competitor), subtle (sample size, leakage, a confound, seed variance, and so on), null (effect is noise, or came from something else you also changed). Give each a prior and its cheapest falsifier (a `Check:` line).
- Run the most discriminating cheap test. Forward-predict each hypothesis ("what would I see if this were the cause?"); a test is strong evidence only where the predictions diverge. Weigh the learning against code and GPU cost.
- Bisect to localize. Data flows forward and gradients flow backward, so probe the midpoint and ask whether the value or gradient is already wrong halfway through. Each probe halves the search space.
- Act only on what the observation pointed to.

## A few non-obvious numbers

The model already recalls most symptom-to-cause pairs, so they don't earn space here. These are the ones it tends to get wrong, worth holding in context:

- Init loss at chance is -ln(1/k) for a k-class softmax (2.30 at k=10). Far above means a loss or init bug; far below suggests data leakage.
- Update-to-data ratio per layer near 1e-3: `((lr*p.grad).std()/p.data.std()).log10()` around -3 [Karpathy].
- Normalize inputs with running stats over all data seen so far; a recent-only window silently shifts the input distribution [Schulman].
- Adam starting LR 3e-4 [Karpathy]. LR scales with batch size, between sqrt and linear for Adam [McCandlish].
- For physics/PINNs, nondimensionalize before training so PDE coefficients are O(1).

## Reference (load when you need the menu)

Each of these is a menu of hypotheses to widen your search, accurate to its own setting, so read the relevant one when the task calls for it rather than up front:

- [Triage decision tree](SKILL.md#63-triage-decision-tree): symptom to first checks, top to bottom.
- [Symptom and gradient tables](SKILL.md#part-1-general-ml-debugging): loss-curve patterns, gradient health, numerical hygiene.
- [Loss-surface and gradient analysis](SKILL.md#part-3-loss-surface--gradient-analysis-no-model-required): visualize a loss before training.
- [Why won't this metric move?](SKILL.md#part-5-diagnosing-why-wont-this-metric-move): structural ceiling vs competing losses.
- [Sweeps and statistics](SKILL.md#part-4-experiment-sweeps--statistical-analysis): same-seed comparisons, within-group z-scores, t-stats.
- [Mental models and priors](SKILL.md#part-7-debugging-folklore--mental-models): five hypothesis-generating lenses, when to suspect data.
- Domain sub-skills: RL [rl/SKILL.md](rl/SKILL.md), PINNs [pinn/SKILL.md](pinn/SKILL.md).

Scope: most sources are 2017-2021. The mindset and loop are durable; specific RL defaults and reward-scaling advice may have moved on. For modern transformer pretraining see [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) and [nanochat deepwiki](https://deepwiki.com/karpathy/nanochat).

## Sources

Claims trace to verbatim quotes in [docs/evidence/](docs/evidence/) via the map in [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown). Starting points: [Karpathy recipe](docs/evidence/karpathy_recipe_training_nn_2019.md), [CS231n](docs/evidence/cs231n_neural_networks_3.md), [Goodfellow Ch11](docs/evidence/goodfellow_ch11_practical_methodology.md), [Schulman Nuts and Bolts](docs/evidence/schulman_nuts_bolts_deeprl_bootcamp_2017_subtitles.md), [Andy Jones](docs/evidence/andyljones_rl_debugging.md).
