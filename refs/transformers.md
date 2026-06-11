# Transformer and LLM debugging folklore

Appendix to the [ML Debugging skill](../SKILL.md). This collects transformer-specific quotes, primary sources, and technical reports; start with the general debugging folklore first.

## Walk and log the full trace

> The best way to debug an error that arises in `trainer.train()` is to manually go through this whole pipeline to see where things went awry.[^hfcourse]

> Debugging a failed run without metrics is guesswork.[^axolotl-stability]

For fine-tuning, inspect decoded tokenized examples and label masks, not just the raw dataset:

> All labels in your dataset are -100. Training losses will be all 0.[^unsloth]

Practical consequence: log the exact rendered prompt, special tokens, system prompt, completion, token IDs, label masks, truncation, generation settings, and model/tokenizer revisions. This follows from the HF pipeline-walkthrough advice, Axolotl's metrics-first guidance, and Unsloth's chat-template/BOS failure cases.

## Match training and deployment

> It's essential to use the SAME chat template that was used when training the model in Unsloth and later when you run it in another framework, such as llama.cpp or Ollama.[^unsloth]

Unsloth also says to test both hypotheses: an unnecessary start-of-sequence token, or a missing one.[^unsloth]

## Warmup and learning rate

> Large-batch training without warmup can diverge in the first epoch and look like a code bug.[^goyal]

Axolotl's SFT stability guide says the learning rate should follow the expected "warmup then decay" schedule, and lists insufficient warmup as a cause of early loss plateaus.[^axolotl-stability] Treat warmup as a strong transformer recipe prior: verify that the LR actually ramps up before the stable/high-LR phase, and that scheduler steps are counted in optimizer steps, not raw microbatches.

> Fine-tuned d12 hyperparameters actively hurt d20 performance.[^nanochat]

Smith and Topin's Super-Convergence paper gives the key empirical support: neural nets trained with "one learning rate cycle and a large maximum learning rate" can train an order of magnitude faster on the workloads they tested.[^super-convergence] Treat this as strong evidence for trying OneCycle, not a universal proof that it is best for every transformer run.

For modern LLM pretraining, also consider WSD (warmup-stable-decay). Wen et al. contrast it with cosine: cosine requires choosing the total step budget up front, while WSD keeps a stable high-LR branch that can be decayed from different checkpoints when the compute budget is known.[^wsd] Warmup can enable an otherwise healthy transformer run; it does not rescue broken labels, masks, data, or gradients. Log the actual LR at every optimizer step and check scheduler units against gradient accumulation.

## Which optimizer?

> In the early stages of setting baselines I like to use Adam with a learning rate of 3e-4. In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate.[^karpathy-recipe]

AdamW remains the robust default. Some modern recipes mix AdamW with matrix-style optimizers for selected parameters, but treat that as a recipe to copy deliberately, not a generic substitution.[^nanochat-optimizer]

There is no context-free winner. A controlled benchmark finds that matrix-based optimizers consistently outperform scalar-based ones, but their speedup over AdamW falls from about `1.4x` at `0.1B` to `1.1x` at `1.2B` parameters.[^optimizer-benchmark]

> Optimal choice of optimizer shifts depends on data-to-model ratios.[^optimizer-benchmark]

That benchmark discusses matrix optimizers such as Muon, Soap, and Kron; the point for debugging is the caveat, not a specific winner. Tune each optimizer fairly, compare at the target scale, batch size, data-to-model ratio, and training budget, and prefer a proven recipe unless optimizer research is the experiment.

The disclosed training reports mostly reinforce this boring answer. DeepSeek-V3, OPT-175B, and Llama 3 all disclose AdamW recipes with warmup and decay; DeepSeek-V3 uses AdamW with a warmup, long stable high-LR phase, cosine decay, late lower-LR phase, gradient clipping, and batch-size scheduling.[^deepseek-v3-report] OPT-175B tried vanilla SGD during divergence recovery; "optimization plateaued quickly," and they reverted to AdamW.[^opt175b-report]

## Better numbers can mean worse learning

> The 'lower validation loss' from BOS-alignment is misleading—it's just fewer noisy tokens, not better learning.[^nanochat]

> Improvements must show gains across multiple axes: per-step efficiency (loss vs. step), wall-clock efficiency (loss vs. time), and compute efficiency (loss vs. FLOPs).[^nanochat]

Inspect the best run's traces. It may have won by learning a shortcut, formatting artifact, or easier token distribution rather than the intended task.

## Distributed and numerical failures

> If any rank's gradient contains inf, all ranks must clip to avoid divergence.[^nanochat]

> As you can see it's the previous frames that we need to look into when the numbers start going into very large for fp16 numbers.[^bekman]

Single-GPU tests can hide distributed failure modes. Keep the frames before a NaN/Inf, not only the crash site.

Modern reports treat infrastructure and numerics as first-class hypotheses, not background. DeepSeek-V3 reports no "irrecoverable loss spikes" or rollbacks after architecture, FP8, high-precision-retention, routing, and schedule co-design.[^deepseek-v3-report] MAI-Thinking-1 says "failures are expected" at thousands of GPUs, and gates nodes through certification before admitting them to production training.[^mai-thinking-report] Llama 3 reports 466 interruptions in a 54-day 405B pretraining window, mostly hardware-related, with automation handling almost all of them.[^llama3-report]

## Is the model too small?

Do not use a universal parameter-count threshold. Chaudhary et al. measure evaluation-awareness probes across 15 models from `0.27B` to `70B` and report predictable scaling rather than a clean threshold.[^eval-awareness] Test a same-family size ladder and separate "the representation is detectable" from "the model can reliably express the behavior."

## Activation steering

> Steering effects are highly variable across samples, and often go in the opposite direction.[^steering-reliability]

The reliability paper finds that higher cosine similarity among training-set activation differences predicts more effective steering.[^steering-reliability] Sweep layers and coefficients, inspect per-example effects, compare against prompting and few-shot baselines, and check whether the vector changed the concept or merely style, verbosity, sentiment, or refusal rate.

## What the recent reports add

OLMo 3 is the strongest "how to decide" reference in this set. It says "benchmarks are not perfect decision-making tools"; small models can sit at random chance, small score differences can be benchmark noise, and some tasks should be expanded, clustered, moved out of averages, or removed.[^olmo3-report] Use proxy metrics and signal-to-noise checks before trusting small-scale ablations.

MAI-Thinking-1 gives the eval-design maxim: "Evaluation results are only as informative as the prompts they are computed on."[^mai-thinking-report] A narrow, saturated, or misweighted eval can give tight confidence intervals around the wrong quantity. Treat eval construction as part of the experiment, not bookkeeping.

Hermes 4 is useful for evaluation reproducibility and reasoning-length control. It says an eval score depends on "the inference engine and hardware" as well as the model, so they route benchmarks through one OpenAI-compatible endpoint and log all evaluation samples.[^hermes4-report] For overlong reasoning, Hermes 4 does a targeted second SFT stage that teaches `</think>` termination without training on the whole generated chain.[^hermes4-report]

Qwen3 is the chat-template and mode-control reminder: thinking/non-thinking behavior is part of the data format, not just sampling policy. Qwen3 uses `/think` and `/no_think` flags and exposes `enable_thinking=False` through the tokenizer chat template.[^qwen3-report]

Hermes 4 and Qwen3 both lean on filtered synthetic/verifiable data, but with guardrails: Hermes uses a different judge model from the answer model to reduce judge self-preference, and Qwen3 filters reasoning traces for wrong answers, repetition, guesswork, thinking/summary inconsistency, style shifts, and possible validation overlap.[^hermes4-report][^qwen3-report]

## Read disclosed-training reports

When debugging or designing a modern transformer run, read reports that disclose the model-building process rather than only final benchmark scores:

- [Olmo 3](https://arxiv.org/abs/2512.13961) releases the "entire model flow," including stages, checkpoints, data, and dependencies; code lives in [OLMo-core](https://github.com/allenai/OLMo-core).
- Microsoft's [MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) treats model development as a system-level optimization problem and gives a long-form account of scaling and RL decisions.
- Nous Research's [Hermes 4](https://arxiv.org/abs/2508.18255) describes failures and solutions across data curation, synthesis, training, and evaluation; Nous also releases open training/evaluation tooling such as [Atropos](https://github.com/NousResearch/atropos).
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) reports architecture, infrastructure, training, and a run with no irrecoverable loss spikes or rollbacks.
- [Qwen3](https://arxiv.org/abs/2505.09388) documents a dense/MoE family from `0.6B` to `235B`, including pretraining and post-training details.
- Secondary postmortems: [The Llama 3 Herd](https://arxiv.org/abs/2407.21783) for large-scale pretraining operations, and [OPT-175B](https://arxiv.org/abs/2205.01068) for training interruptions, instability, and mid-flight recovery.

These are useful as working implementations and experiment logs: copy proven priors, compare the exact computation graph and recipe, and look for engineering details absent from method papers.

For experiment design, keep the [Google Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook) nearby: it is explicitly about the practical gap between superficially similar recipes and actually working deep-learning systems.[^tuning-playbook]

## Sources

[^hfcourse]: Hugging Face LLM Course, ["Debugging the training pipeline"](https://huggingface.co/learn/llm-course/chapter8/4) ([cache](../docs/evidence/hf_llm_course_ch8_4_debugging_pipeline.md))
[^axolotl-stability]: Axolotl, ["Training Stability"](https://docs.axolotl.ai/docs/training_stability.html) ([cache](../docs/evidence/axolotl_training_stability.md))
[^unsloth]: Unsloth, ["Troubleshooting & FAQs"](https://docs.unsloth.ai/basics/troubleshooting-and-faqs) ([cache](../docs/evidence/unsloth_troubleshooting_faqs.md))
[^goyal]: Goyal et al., ["Accurate, Large Minibatch SGD"](https://arxiv.org/abs/1706.02677)
[^super-convergence]: Smith and Topin, ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"](https://arxiv.org/abs/1708.07120)
[^wsd]: Wen et al., ["Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective"](https://arxiv.org/abs/2410.05192)
[^nanochat]: Karpathy, [nanochat experiment log](https://github.com/karpathy/nanochat/blob/main/dev/LOG.md) ([cache](../docs/evidence/nanochat_deepwiki_llm_pretraining_2026.md))
[^karpathy-recipe]: Karpathy, ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/) ([cache](../docs/evidence/karpathy_recipe_training_nn_2019.md))
[^nanochat-optimizer]: Karpathy, [`nanochat`](https://github.com/karpathy/nanochat) (`optim.py`: AdamW + Muon)
[^optimizer-benchmark]: Wen et al., ["Fantastic Pretraining Optimizers and Where to Find Them"](https://arxiv.org/abs/2509.02046) (ICLR 2026)
[^tuning-playbook]: Google Developers, ["Deep Learning Tuning Playbook"](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook)
[^bekman]: Stas Bekman, [`DebugUnderflowOverflow`](https://github.com/huggingface/transformers/blob/main/src/transformers/debug_utils.py) ([cache](../docs/evidence/bekman_debug_utils_transformers.md))
[^eval-awareness]: Chaudhary et al., ["Evaluation Awareness Scales Predictably in Open-Weights Large Language Models"](https://arxiv.org/abs/2509.13333)
[^steering-reliability]: Braun et al., ["Understanding (Un)Reliability of Steering Vectors in Language Models"](https://arxiv.org/abs/2505.22637)
[^olmo3-report]: OLMo Team, ["Olmo 3"](https://arxiv.org/abs/2512.13961) ([cache](../docs/evidence/reports/olmo3_technical_report.md); [OLMo-core](https://github.com/allenai/OLMo-core), [cache](../docs/evidence/reports/code/olmo_core_readme.md))
[^mai-thinking-report]: Microsoft AI Team, ["MAI-Thinking-1: Building a Hill-Climbing Machine"](https://microsoft.ai/pdf/mai-thinking-1.pdf) ([cache](../docs/evidence/reports/mai_thinking_1_technical_report.md))
[^hermes4-report]: Nous Research, ["Hermes 4 Technical Report"](https://arxiv.org/abs/2508.18255) ([cache](../docs/evidence/reports/hermes4_technical_report.md); [Atropos](https://github.com/NousResearch/atropos), [cache](../docs/evidence/reports/code/nous_atropos_readme.md))
[^deepseek-v3-report]: DeepSeek-AI, ["DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) ([cache](../docs/evidence/reports/deepseek_v3_technical_report.md))
[^qwen3-report]: Qwen Team, ["Qwen3 Technical Report"](https://arxiv.org/abs/2505.09388) ([cache](../docs/evidence/reports/qwen3_technical_report.md))
[^llama3-report]: Meta AI, ["The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783) ([cache](../docs/evidence/reports/llama3_herd_technical_report.md))
[^opt175b-report]: Zhang et al., ["OPT: Open Pre-trained Transformer Language Models"](https://arxiv.org/abs/2205.01068) ([cache](../docs/evidence/reports/opt175b_technical_report.md))
