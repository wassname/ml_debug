---
name: ml-debug
description: "Wassname's practical folklore for debugging ML systems: convergence, gradients, stuck metrics, sweep reliability, same-seed comparisons. Use when stuck on training, designing sweeps, or analyzing results. Verbatim quotes from practitioners; catalogs and diagnostics are one hop away."
---

# wassname's ML Debugging Folklore

Foreword: In an attempt to upskill the ML debugging on AI coding assistants (and humans), I've collected high quality sources on ML debugging and the mindset and the "taste". When I started ML I went searching for discussions on best practices, and started a few discussions of my own and they helped me a lot, I hope they can help others. This intro is human written, and the below is AI written with human guidance.

## How to read this

> Wassname's debugging loop (unpublished): write at least three possible worlds before acting: the most likely failure, a subtle failure, a perverse failure, a possible bug, and an unknown if relevant. Put a rough credence/prior on each. For each world, say what you expect to see differently and the cheapest evidence that would distinguish it.

If you're an LLM agent, calibrate yourself first. ML research code is often outside your training distribution: novel losses, custom architectures, methods with no canonical right answer you've seen a thousand times. The trained reflex there is to be confident and fast, to pattern-match a symptom to a fix ("loss stuck -> drop the LR") and apply it, and on possibly-buggy research code that reflex wastes a run and corrupts the evidence you need to find the real cause. The quotes below are the counter-evidence, in the words of people who paid for these lessons in months of wasted runs. Before acting: form competing hypotheses and identify evidence that distinguishes them ([Rahtz](#think-more-experiment-less)); assume a correctness bug before tuning ([Jones](#assume-you-have-a-bug)); instrument silent failure paths and test more than one setup ([Achiam](#broken-code-fails-silently-measure-everything-spinning-up)); inspect the data and seek falsifiers before believing the result ([Nanda](#default-to-disbelieving-your-own-results-neel-nanda)).

These are common failure modes worth ruling out, not a complete diagnosis of your situation; you know your system and I don't. Checklists, diagnostics, and symptom catalogs are one hop away under [Reference](#reference-one-hop-away). The short version of Rahtz plus the tuning playbook is: compare at least three possible worlds, put rough credences on them, include a bug and an unknown if relevant, predict what evidence differs between them, then run the narrowest experiment that can actually distinguish them.

## Folklore

### Assume you have a bug

> When their RL implementation doesn't work, people are often keen to either (a) adjust their network architecture or (b) adjust their hyperparameters. On the other hand, they're reluctant to say they've got a bug. Most often, it turns out they've got a bug. Why bugs are so much more common in RL code is discussed above, but there's another advantage to assuming you've got a bug: bugs are a damn sight faster to find and fix than validating that your new architecture is an improvement over the old one.[^jones]

> What I'm advocating for here is not a blind faith in the buginess of your code, but for dramatically raising the threshold at which you start thinking 'OK, I think this is correct.'[^jones]

A bug can also hide, because most ML models have multiple adaptive parts: "If one part is broken, the other parts can adapt and still achieve roughly acceptable performance"[^goodfellow], and it may not show in the output at all.

### Think more, experiment less

> Switching from experimenting a lot and thinking a little to experimenting a little and thinking a lot was a key turnaround in productivity. When debugging with long iteration times, you really need to *pour* time into the hypothesis-forming step - thinking about what all the possibilities are, how likely they seem on their own, and how likely they seem in light of everything you've seen so far. Spend as much time as you need, even if it takes 30 minutes, or an hour. Reserve experiments for once you've fleshed out the hypothesis space as thoroughly as possible and know which pieces of evidence would allow you to best distinguish between the different possibilities.[^rahtz]

### Don't write RL from scratch; diff against a reference

> If you're doing anything that involves an RL algorithm as a component in a larger system, don't try and implement the RL algorithm yourself. [...] RL is unstable enough at the moment that you'll never be sure whether your system doesn't work because of a bug in your RL implementation or because of a bug in your larger system.[^rahtz]

> We find that implementation differences which are often not reflected in publications can have dramatic impacts on performance.[^henderson]

When you're stuck after a diagnostic cycle or two, the generalization of this advice is to find a working implementation (rank candidates by community adoption > papers citing it > code that runs > author reputation) and diff your math, computation graph, and hyperparameters against it. For RL see [rl/SKILL.md](rl/SKILL.md).

### Default to disbelieving your own results (Neel Nanda)

> The default state of the world is that your research is false, because doing research is hard.[^nanda]

> Excitement is evidence of bullshit: Generally, most true results are not exciting, but a fair amount of false results are. So from a Bayesian perspective, if a result is exciting and cool, it's even more likely to be false than normal![^nanda]

The cheapest antidote he gives: "Read your data ... Often, the quality of the data is a crucial driver of the results of your experiments. Often, it is quite bad."[^nanda]

### Understand the system to shrink the search (Ulisse Mini)

> When good programmers debug hard problems fast, it's usually because they understand the system well enough to *track the important internal state* in their head, letting them drastically *reduce the solution space they're searching over.*[^ulisse]

### Gears beat black boxes (John Wentworth)

> figuring out a system's gears takes extra work up-front, but yields dividends forever. [...] The black-box approach is cheaper for one-off tasks, but usually doesn't yield any insights which will generalize to new tasks using the same system[^wentworth]


### Broken code fails silently; measure everything (Spinning Up)

Josh Achiam's warning is RL-framed but general:

> broken RL code almost always fails silently, where the code appears to run fine except that the agent never learns how to solve the task.[^spinningup]

So instrument heavily, because "you can't tell it's broken if you can't see that it's breaking,"[^spinningup] and don't trust one passing setup: "sometimes things will work in one environment even when you have a breaking bug, so make sure to test in more than one environment."[^spinningup]

### Pursue anomalies; investigate confusion

> If you ever see a plot or a behaviour that just *seems weird*, chase right after it! Do not - do *not* - just 'hope it goes away'. Chasing anomalies is one of the most powerful ways to debug your system, because if you've noticed a problem without having had to go look for it, that means it's a *really big problem*. [...] It's really tempting to think that the cool extra functionality you were planning to write today [...] might just magically fix this anomalous behaviour. It won't. Give up on your plan for the day and chase the anomaly instead.[^jones]

> It was only by following that confusion and realising that taking the difference between frames zeroed out the background that gave the hint of a problem with normalization.[^rahtz]
>
> It seems important to really commit yourself to *always* investigate whenever you notice confusion.[^rahtz]

### Read what you actually wrote, not what you meant (gwern)

You can't see your own work clearly, which is why fresh eyes (or a fresh-eyes subagent) catch what you can't:

> you can't find typos in your own writing without a great deal of effort because you know what it's *supposed* to say; so copyediting advice runs like 'read it out loud' or 'print it out and read it' or 'wait a week' [...] or even 'read it upside down'. That's the sort of thing it takes to force you to read what you actually wrote, and not what you thought you wrote.[^gwern-unseeing]

### Never accept the kludge (Patrick Kidger)

Why is research code so reliably buggy? Kidger's blunt answer:

> Academic software is almost always a poorly-maintained kludge of leaky abstractions, awful formatting, and bugs that don't cripple things only because some other bug stops them from doing so.[^kidger]

> This is a systemic professional failing. [...] the overwhelming majority of your time will be spent in front of a screen, staring at code. And yet most of you (yes, you) would not pass muster as a junior developer.[^kidger]

His fix is a posture, "never accept the kludge": messed up your git repo? Find the commands to fix it, "don't just delete it and clone from the remote."[^kidger] The instinct that refuses kludges is the same one that refuses `.detach()`-to-silence-autograd and `except: pass`.

### Loss curves are a red herring

> When someone's RL implementation isn't working, they *luuuuuurv* to copy-paste a screenshot of their loss curve to you. They do this because they know they want a pretty, exponentially-decaying loss curve, and they know what they have *isn't that*. The problem with using the loss curve as an indicator of correctness is somewhat that it's not reliable, but mostly because it doesn't localise errors. The shape of your loss curve says very little about where in your code you've messed up, and so says very little about what you need to change to get things working.[^jones]

(But sometimes they are not, they seperate underfitting and over, gradient explosion vs vanishing, saturation vs not... and so on)

### Inspect the data first

> The first step to training a neural net is to not touch any neural net code at all and instead begin by thoroughly inspecting your data. [...] The outliers especially almost always uncover some bugs in data quality or preprocessing.[^karpathy-recipe]

Slavv's "37 reasons" list opens with the same anecdote (gradients flowing, loss falling, predictions all background) and puts "Verify that the input data is correct" and "Start with a really small dataset (2-20 samples). Overfit on it" at the top of its emergency checklist[^slavv].

Andrew Ng's error-analysis procedure is the same move applied after your first trained model: before investing a month in any fix, gather ~100 misclassified dev examples and count the failure categories in a spreadsheet.

> Manually examining 100 examples does not take long. Even if you take one minute per image, you'd be done in under two hours. These two hours could save you a month of wasted effort.[^ng-mly]

### Labels are often wrong (koaning)

Even benchmark data is dirtier than you think. Vincent Warmerdam:

> It turns out that bad labels are a *huge* problem in many popular benchmark datasets.[^koaning]

His cheap way to find them: train a deliberately high-bias model, then sort by where it disagrees with the label while assigning the correct class low confidence. The takeaway: "maybe we should spend [...] less time tuning parameters and instead spend it trying to get a more meaningful dataset."[^koaning]

### The tank story: your model learns the confound (gwern)

The canonical data-leakage parable:

> A cautionary tale in artificial intelligence tells about researchers training an neural network (NN) to detect tanks in photographs, succeeding, only to realize the photographs had been collected under specific conditions for tanks/non-tanks and the NN had learned something useless like time of day.[^gwern]

gwern traced versions back to 1992 and concluded it is "a classic 'urban legend'" with no solid source[^gwern]. The lesson holds twice over: a model will gladly learn a confound in how the data was collected instead of the task, and even your cautionary tales deserve a citation.

### Test-set contamination is insidious (Domingos)

Domingos' 2012 CACM paper set out to write down ML "folk knowledge" (the same project as this file):

> Doing well on the training set is easy (just memorize the examples). The most common mistake among machine learning beginners is to test on the training data and have the illusion of success.[^domingos]

> Contamination of your classifier by test data can occur in insidious ways, for example, if you use test data to tune parameters and do a lot of tuning. (Machine learning algorithms have lots of knobs, and success often comes from twiddling them a lot, so this is a real concern.)[^domingos]

Lones catalogs the concrete leak routes: scaling statistics computed on the full dataset before splitting, augmentation before splitting, look-ahead bias when cross-validating time series[^lones].

### Overfit one batch first

> Overfit a tiny subset of data. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it's also best to set regularization to zero [...]. Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset.[^cs231n]

> Overfit a single batch of only a few examples (e.g. as little as two). [...] If they do not, there is a bug somewhere and we cannot continue to the next stage.[^karpathy-recipe]

And remove a variable while you're at it: "Always use a fixed random seed [...]. This removes a factor of variation and will help keep you sane."[^karpathy-recipe]

### The most common neural net mistakes (Karpathy)

The 2018 tweet thread that seeded the recipe post. Every item is a silent failure except 5:

> most common neural net mistakes: 1) you didn't try to overfit a single batch first. 2) you forgot to toggle train/eval mode for the net. 3) you forgot to .zero_grad() (in pytorch) before .backward(). 4) you passed softmaxed outputs to a loss that expects raw logits. ; others? :)[^karpathy-mistakes]

> oh: 5) you didn't use bias=False for your Linear/Conv2d layer when using BatchNorm, or conversely forget to include it for the output layer .This one won't make you silently fail, but they are spurious parameters[^karpathy-mistakes]

> 6) thinking view() and permute() are the same thing (& incorrectly using view)[^karpathy-mistakes]

Number 6 is the bug the backprop-to-input dependency check catches mechanically ([refs/diagnostics.md](refs/diagnostics.md)).

### Seed variance: you can't tell a bug from bad luck

> Look, there's variance in supervised learning too, but it's rarely this bad. If my supervised learning code failed to beat random chance 30% of the time, I'd have super high confidence there was a bug in data loading or training. If my reinforcement learning code does no better than random, I have no idea if it's a bug, if my hyperparameters are bad, or if I simply got unlucky.[^irpan]

> Instability to random seed is like a canary in a coal mine. If pure randomness is enough to lead to this much variance between runs, imagine how much an actual difference in the code could make.[^irpan]

Henderson confirmed it quantitatively: splitting 10 same-config runs (differing only in seed) into two groups of five produces "statistically different distributions just from varying random seeds."[^henderson] This is why one good run proves nothing ([refs/sweeps.md](refs/sweeps.md)).

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

Many normalization/regularization tricks do roughly the same job (they improve conditioning), so stacking them adds complexity without proportional benefit.

### Changing anything changes everything (Sculley et al.)

Why ablation and one-change-at-a-time work, from Google's production-ML technical-debt paper:

> **Entanglement.** Machine learning systems mix signals together, entangling them and making isolation of improvements impossible. For instance, consider a system that uses features x1, ...xn in a model. If we change the input distribution of values in x1, the importance, weights, or use of the remaining n − 1 features may all change. [...] No inputs are ever really independent. We refer to this here as the CACE principle: Changing Anything Changes Everything. CACE applies not only to input signals, but also to hyper-parameters, learning settings, sampling methods, convergence thresholds, data selection, and essentially every other possible tweak.[^sculley]

This is also why "I changed the method and a hyperparameter and it got better" tells you nothing about the method.

### Exploration over exploitation (Google tuning playbook)

The Google Research tuning playbook opens by admitting there is "an astonishing amount of toil and guesswork" in getting deep nets to work; their counter is experiment-design discipline:

> Although one might think we would spend most of our time trying to maximize performance on the validation set, in practice we spend the majority of our time trying to gain insight into the problem, and comparatively little time greedily focused on the validation error. In other words, we spend most of our time on "exploration" and only a small amount on "exploitation".[^tuning-playbook]

Their experiment-design vocabulary is the reusable part: each round has *scientific* hyperparameters (the thing you're measuring), *nuisance* hyperparameters (must be re-tuned for the comparison to be fair), and *fixed* ones (caveats on your conclusions).

> The learning rate is a nuisance hyperparameter because we can only fairly compare models with different numbers of hidden layers if the learning rate is tuned separately for each number of layers (the optimal learning rate generally depends on the model architecture).[^tuning-playbook]

### Adam at 3e-4 for baselines (Karpathy)

> In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate.[^karpathy-recipe]

If you change the batch size, the learning rate has to move with it: linearly for SGD[^goyal], with an exponent between 0.5 and 1 for Adam[^mccandlish], and large-batch training without warmup can diverge in the first epoch and look like a code bug[^goyal].

## Modern transformers and LLM fine-tuning

Most of the sources above predate large transformers; these come from the people training and fine-tuning them.

### Tricks hide in reference code (lucidrains)

lucidrains' x-transformers is a catalogue of training tricks, each tied to its paper. The debugging-relevant one: when a transformer diverges, attention logits blowing up is a prime suspect, and the now-standard fix is QK normalization.

> We are nearing the point of wiping out a source of transformer training instability with one simple intervention.[^lucidrains]

Scaled-up recipes accumulate these one-line stability fixes in code long before they're written up, which is the whole case for reading a working implementation.

### Modern LLM-pretraining gotchas (nanochat)

Karpathy's nanochat is one of the few public records of what scaling a transformer from scratch actually takes. Two gotchas worth stealing:

> The 'lower validation loss' from BOS-alignment is misleading—it's just fewer noisy tokens, not better learning.[^nanochat]

> If any rank's gradient contains inf, all ranks must clip to avoid divergence.[^nanochat]

The first is a better number that isn't better learning; the second is a multi-GPU bug that single-GPU testing hides.

### When NaN hits, look at the frames before it (Stas Bekman)

Bekman wrote the `DebugUnderflowOverflow` tool during BLOOM-era large-model training. It keeps a rolling buffer of per-module abs-min/abs-max frames, so when inf/NaN is detected you see the run-up rather than only the crash site.

> As you can see it's the previous frames that we need to look into when the numbers start going into very large for fp16 numbers.[^bekman]

Corollary from the same docstring: validate your debugging instrumentation on a few cheap batches before betting an hours-long run on it.

### Loss spikes usually mean a bad data pocket (Stas Bekman)

Bekman's ML Engineering book has a gallery of real loss-curve pathologies from BLOOM and IDEFICS training, with the honest caveat that "very often we don't really understand why certain types of spikes happen" and pattern recognition is the realistic goal:

> In general there are 3 types of loss spikes: 1. Fast recovering spikes 2. Slow recovering spikes 3. Not fully recovering spikes
>
> The spikes usually happen because of a bad data pocket, either due to badly shuffled data or because it hasn't been cleaned from some garbage scraped from the websites.[^bekman-book]

And the post-mortem of the 104B model that diverged for months before BLOOM-176B succeeded:

> We think the 2 main obstacles were using fp16 and data that had a lot of garbage in it. For BLOOM-176B we switched to bf16, used much cleaner data and also added an embedding layer-norm and that made all the difference.[^bekman-book]

His recommended way to build this intuition: "The best learning is to read Publicly available training LLM/VLM logbooks because there you can see exactly what happened and how the problem has been overcome."[^bekman-book]

### Walk the pipeline in data order (HF course)

The HF LLM course debugging chapter is a worked narrative in the Karpathy-recipe lineage: a deliberately broken fine-tune, fixed step by step, checking each stage at the exact point it enters the model.

> The best way to debug an error that arises in `trainer.train()` is to manually go through this whole pipeline to see where things went awry. The error is then often very easy to solve.[^hfcourse]

> Hyperparameter tuning is always emphasized as being the hardest part of machine learning, but it's just the last step to help you gain a little bit on the metric. [...] don't launch into a time-consuming and costly hyperparameter search until you have something that beats the baseline you have on your dataset.[^hfcourse]

### Chat template and BOS handling must match across train and deploy (unsloth)

When a model trains fine but produces nonsense after export to llama.cpp or Ollama, the weights are usually innocent:

> The most common cause of this error is using an **incorrect chat template**. It's essential to use the SAME chat template that was used when training the model in Unsloth and later when you run it in another framework, such as llama.cpp or Ollama. [...] It might also be because your inference engine adds an unnecessary "start of sequence" token (or the lack of thereof on the contrary) so ensure you check both hypotheses![^unsloth]

Their FAQ also explains the suspiciously perfect loss curve: when the loss sits at exactly zero, every label has probably been masked out and the model is learning nothing.

> All labels in your dataset are -100. Training losses will be all 0.[^unsloth]

### Shrink every axis at once, and clear the caches (axolotl)

Axolotl's debugging guide (the general tips trace to Hamel Husain) gives the minimal-repro recipe for training loops: one GPU, one process, a tiny model, tiny data, a single step, no eval. It also warns that caching can quietly undo your experiment, because the run you think you changed may be replaying artifacts produced before the change:

> **Eliminate concurrency**: Restrict the number of processes to 1 for both training and data preprocessing[^axolotl]

> Axolotl caches certain steps and so does the underlying HuggingFace trainer. You may want to clear some of these caches when debugging.[^axolotl]

Their training-stability page adds the masking check ("inspect tokenized samples to confirm only the target tokens are trainable") and, bluntly: "Debugging a failed run without metrics is guesswork."[^axolotl-stability]

## Reference (one hop away)

Open the relevant one when the task calls for it. These are synthesized checklists and menus, useful for widening a hypothesis search but not authoritative for your particular system:

- [PLAYBOOK.md](PLAYBOOK.md) — the long-form version: mental models and practitioner priors, the general step catalog (component isolation, baseline ladder, what to log, numerical hygiene), symptom tables, the agent debugging loop, triage, and anti-patterns.
- [refs/checklist.md](refs/checklist.md) — Lones's full 36-item do/don't checklist across data, building, evaluation, comparison, and reporting.
- [refs/diagnostics.md](refs/diagnostics.md) — copy-paste diagnostic snippets: init-loss check, overfit-one-batch, gradient-flow check, NaN hooks, NaN-poisoning leakage tracer, backprop-to-input dependency check, class-imbalance check.
- [refs/static_analysis.md](refs/static_analysis.md) — grep patterns for silent bugs (shape mismatches, autograd breakers, double softmax, step ordering, leakage).
- [refs/loss_surface.md](refs/loss_surface.md) — visualize a loss surface and its gradient field with synthetic tensors, no model or GPU, for when a custom loss misbehaves.
- [refs/metric_stuck.md](refs/metric_stuck.md) — "why won't this metric move?" plus the structural-ceiling check.
- [refs/sweeps.md](refs/sweeps.md) — same-seed paired comparison and cross-seed t-stat reliability, for before you claim method A beats method B.
- [refs/llm_judges.md](refs/llm_judges.md) — LLM-as-a-judge biases (position, verbosity, self-preference) and the mitigation checklist, for when an LLM-judged eval looks too good.
- [refs/transformers.md](refs/transformers.md) — transformer-specific folklore: full traces, warmup/LR, optimizer evidence, train-deploy parity, scale priors, steering, and disclosed-training reports.
- [rl/SKILL.md](rl/SKILL.md) — RL-specific: probe environments, reward engineering, HP defaults, reference implementations.
- [pinn/SKILL.md](pinn/SKILL.md) — physics-informed networks: nondimensionalization, gradient pathologies, curriculum.

## Links and further reading

Start here rather than treating the bibliography as flat:

- **Beginner / broad checklist:** Lones, ["How to avoid machine learning pitfalls"](https://arxiv.org/abs/2108.02497), with its full do/don't list extracted in [refs/checklist.md](refs/checklist.md).
- **Debugging a neural net:** Karpathy, ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/).
- **Designing tuning experiments:** Google, [Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook).
- **Transformer and LLM runs:** [refs/transformers.md](refs/transformers.md), then the HF, Axolotl, Unsloth, nanochat, and Bekman sources below.

Folklore sources (the quotes above trace to these):

[^jones]: Andy Jones, "Debugging RL, Without the Agonizing Pain" — https://andyljones.com/posts/rl-debugging.html ([cache](docs/evidence/andyljones_rl_debugging.md): anomalies L103-109, write-from-scratch L155, assume-bug L176-180, raise-threshold L182, loss-curve L186-188)
[^rahtz]: Matthew Rahtz (Amid Fish), "Lessons Learned Reproducing a Deep RL Paper" — http://amid.fish/reproducing-deep-rl ([cache](docs/evidence/amid_fish_reproducing_deep_rl.md): frame-diff confusion L85-87, investigate-confusion L100-102, think-more L145-153, don't-implement-RL-yourself L497-501)
[^karpathy-recipe]: Andrej Karpathy, "A Recipe for Training Neural Networks" (2019) — https://karpathy.github.io/2019/04/25/recipe/ ([cache](docs/evidence/karpathy_recipe_training_nn_2019.md): inspect-data L26+L32, fixed-seed L39, overfit-one-batch L51, Adam-3e-4 L73; note: this is an abridged note with its own "..." elisions)
[^karpathy-mistakes]: Andrej Karpathy, "most common neural net mistakes" tweet thread, 1 Jul 2018 — https://x.com/karpathy/status/1013244313327681536 ([cache](docs/evidence/karpathy_common_mistakes_tweet_2018.md): tweets 1-3 verbatim, cross-checked against threadreaderapp; x.com itself blocks fetching)
[^sculley]: Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015) — https://papers.nips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf ([cache](docs/evidence/sculley_2015_hidden_technical_debt.md): abstract, CACE/entanglement, ensemble caveat)
[^schulman]: John Schulman, "Nuts and Bolts of Deep RL Research" slides — http://joschu.net/docs/nuts-and-bolts.pdf ([cache](docs/evidence/joschu_nuts_and_bolts.md): Always-Be-Ablating L98-101, standardize-observations L118-125; rendered as bullets because the PDF source is slide fragments)
[^henderson]: Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018) — https://arxiv.org/abs/1709.06560 ([cache](docs/evidence/henderson_2018_deep_rl_matters.md): seeds-create-different-distributions L235, implementation-differences L251)
[^irpan]: Alex Irpan, "Deep Reinforcement Learning Doesn't Work Yet" (2018) — https://www.alexirpan.com/2018/02/14/rl-hard.html ([cache](docs/evidence/alexirpan_rl_hard.md): variance-bug-or-unlucky L674-678, seed-canary L705-707)
[^cs231n]: Stanford CS231n, "Neural Networks Part 3" — https://cs231n.github.io/neural-networks-3/ ([cache](docs/evidence/cs231n_neural_networks_3.md): overfit-tiny-subset L89)
[^slavv]: Slav Ivanov, "37 Reasons why your Neural Network is not working" (2017) — https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607 ([cache](docs/evidence/slavv_37_reasons_nn.md): opening anecdote L19, emergency checklist L45-51)
[^goodfellow]: Goodfellow, Bengio, Courville, *Deep Learning*, ch. 11 "Practical Methodology" — https://www.deeplearningbook.org/ ([cache](docs/evidence/goodfellow_ch11_practical_methodology.md): one-part-broken-others-adapt L198, weights-adapt-to-compensate L204)
[^mccandlish]: McCandlish, Kaplan et al., "An Empirical Model of Large-Batch Training" (2018) — https://arxiv.org/abs/1812.06162 ([cache](docs/evidence/mccandlish_2018_large_batch.md))
[^goyal]: Goyal et al., "Accurate, Large Minibatch SGD" (2017) — https://arxiv.org/abs/1706.02677
[^lucidrains]: Phil Wang (lucidrains), x-transformers README — https://github.com/lucidrains/x-transformers ([cache](docs/evidence/lucidrains_x_transformers_readme.md): post-embedding LayerNorm / BLOOM+YaLM L366, attention-overflow / cosine-sim norm L1230, autoregressive validation L1234, "wiping out a source of instability" / QK RMSNorm L1292)
[^koaning]: Vincent D. Warmerdam (koaning), "Bad Labels" (2021) — https://koaning.io/posts/labels/ ([cache](docs/evidence/koaning_bad_labels.md): bad-labels-huge-problem L13, confidence-sort trick L21, spend-less-time-tuning L33)
[^nanochat]: nanochat (Karpathy), documented via DeepWiki — https://deepwiki.com/karpathy/nanochat ([cache](docs/evidence/nanochat_deepwiki_llm_pretraining_2026.md): BOS fake-improvement L97, all-ranks-clip-on-inf L131)
[^kidger]: Patrick Kidger, "Just Know Stuff" (2023) — https://kidger.site/thoughts/just-know-stuff/ ([cache](docs/evidence/kidger_just_know_stuff.md): kludge-definition L7, junior-developer L9, never-accept-the-kludge L11, don't-delete-and-clone L13)
[^gwern]: Gwern Branwen, "The Neural Net Tank Legend" — https://gwern.net/tank ([cache](docs/evidence/gwern_tank.md): cautionary tale L7, urban-legend conclusion L9)
[^spinningup]: Joshua Achiam, "Spinning Up as a Deep RL Researcher" (OpenAI, 2018) — https://spinningup.openai.com/en/latest/spinningup/spinningup.html ([cache](docs/evidence/spinningup_researcher.md): fails-silently L11, test-more-than-one-env L19, measure-everything L21)
[^nanda]: Neel Nanda, "How to Become a Mechanistic Interpretability Researcher" — https://www.alignmentforum.org/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher ([cache](docs/evidence/nanda_how_to_mech_interp.md): research-is-false L7, excitement-is-bullshit L9, read-your-data L11)
[^gwern-unseeing]: Gwern Branwen, "Unseeing" — https://gwern.net/unseeing ([cache](docs/evidence/gwern_unseeing.md): read-what-you-wrote L9, single-anomaly L13)
[^ulisse]: Ulisse Mini, "How to get good at programming" — https://www.lesswrong.com/posts/LTypqBMTSmRrrhb2v/how-to-get-good-at-programming ([cache](docs/evidence/ulisse_how_to_get_good_at_programming.md): track-internal-state L7, brute-force-search L9, leaky-abstractions L11)
[^wentworth]: John Wentworth, "Gears-Level Models are Capital Investments" — https://www.lesswrong.com/posts/nEBbw2Bc2CnN2RMxy/gears-level-models-are-capital-investments ([cache](docs/evidence/wentworth_gears_level_models.md): gears-dividends L7, valley-of-bad-theory L11)
[^hfcourse]: Sylvain Gugger et al., HF LLM Course ch. 8.4, "Debugging the training pipeline" — https://huggingface.co/learn/llm-course/chapter8/4 ([cache](docs/evidence/hf_llm_course_ch8_4_debugging_pipeline.md): walk-the-pipeline L14, overfit-one-batch L678-680, no-tuning-before-baseline L724-726)
[^bekman]: Stas Bekman, `DebugUnderflowOverflow` docstring, transformers `debug_utils.py` (2021) — https://github.com/huggingface/transformers/blob/main/src/transformers/debug_utils.py ([cache](docs/evidence/bekman_debug_utils_transformers.md): purpose L35-36, detection-and-frame-buffer L51-53, previous-frames L86-92)
[^unsloth]: Unsloth (Daniel & Michael Han-Chen), "Troubleshooting & FAQs" — https://docs.unsloth.ai/basics/troubleshooting-and-faqs ([cache](docs/evidence/unsloth_troubleshooting_faqs.md): template-mismatch + BOS L38-39, shuffle-eval L100, all-labels–100-loss-0 L227-229)
[^axolotl]: Axolotl, "Debugging" (general tips: Hamel Husain) — https://docs.axolotl.ai/docs/debugging.html ([cache](docs/evidence/axolotl_debugging.md): simplify L31, one-process L37, small-model + fast-iteration L48-49, caches L54-58)
[^axolotl-stability]: Axolotl, "Training Stability" — https://docs.axolotl.ai/docs/training_stability.html ([cache](docs/evidence/axolotl_training_stability.md): metrics-from-the-start L27, inspect-tokenized-masking L67, reward-fn-standalone L99)
[^ng-mly]: Andrew Ng, *Machine Learning Yearning* (2018 draft), ch. 13-19 on error analysis — https://github.com/ajaymache/machine-learning-yearning ([cache](docs/evidence/ng_ml_yearning_error_analysis.md): build-first-system L10, 100-examples procedure L14-20, Eyeball/Blackbox dev sets L32)
[^tuning-playbook]: Godbole, Dahl, Gilmer, Shallue, Nado, "Deep Learning Tuning Playbook" (Google Research / Google Developers, 2023; Google Developers page last updated 2025-08-25) — https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook ([cache](docs/evidence/google_tuning_playbook.md): exploration-over-exploitation L24, scientific/nuisance/fixed L34-38, incremental-tuning L14-18)
[^domingos]: Pedro Domingos, "A Few Useful Things to Know About Machine Learning" (CACM, Oct 2012) — https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf ([cache](docs/evidence/domingos_2012_few_useful_things.md): test-on-train illusion L20, insidious-contamination L22, overfitting-bugbear L26, features-are-key L32)
[^bekman-book]: Stas Bekman, *Machine Learning Engineering Open Book*, "Understanding Training Loss Patterns" + "Instabilities" — https://github.com/stas00/ml-engineering ([cache](docs/evidence/bekman_ml_engineering_instabilities.md): heartbeat L10, 104B post-mortem L18, spike types + bad-data-pocket L22-24, init-std L28-32, PaLM batch-skipping L36, logbooks L40)
[^lones]: Michael A. Lones, "How to avoid machine learning pitfalls" (2021, updated annually) — https://arxiv.org/abs/2108.02497 ([cache](docs/evidence/lones_2021_ml_pitfalls.md): full do/don't TOC L18-22, leakage L26, look-ahead bias L30). Aimed at beginners but the most exhaustive checklist here: 36 do/don'ts across data prep, training, evaluation, comparison, and reporting.

For modern transformer pretraining specifically (most sources above predate it), see [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/) and the [nanochat deepwiki](https://deepwiki.com/karpathy/nanochat) (320+ empirical HP sweeps for a GPT-2-scale run). For LLM-as-judge eval debugging workflow more broadly, Hamel Husain's ["Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/) covers the error-analysis-first approach for LLM products. Most multi-source claims trace to quotes in [docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown) (vargdown); the full evidence set is in [docs/evidence/](docs/evidence/).

Curated by [wassname](https://github.com/wassname).
