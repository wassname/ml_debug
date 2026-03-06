# ML Debugging Folklore - Vargdown Process Log

## Process
- [x] evidence files read (21 files, 9416 lines total)
- [x] quotes extracted via 12 parallel subagents
- [x] key quotes verified against evidence files (spot-checked ~15 quotes)
- [x] argdown verifier passes clean (`npx @argdown/cli json` -- 14 arguments, 45 statements, 14 relations)
- [x] subagent review done (gpt-5.2-codex via opencode; fixed non-verbatim quotes, credence calibration, PCS structure)
- [ ] human review done

## Evidence Fetch Log

All evidence files were pre-existing in `docs/evidence/`. They were fetched
in a prior session via the methods listed in each file's header.

| Source | Evidence File | Fetch Method | Status |
|--------|--------|--------|--------|
| Schulman 2016 slides | joschu_nuts_and_bolts.md | `uvx markitdown[pdf]` | verbatim (PDF artifacts: cid markers) |
| Schulman 2017 bootcamp | schulman_nuts_bolts_deeprl_bootcamp_2017_subtitles.md | YouTube auto-subtitles | verbatim (transcription errors: "insanity" = "and standard") |
| Andy Jones RL debugging | andyljones_rl_debugging.md | markitdown | verbatim |
| Henderson et al. 2018 | henderson_2018_deep_rl_matters.md | markitdown | verbatim |
| Goodfellow Ch11 | goodfellow_ch11_practical_methodology.md | markitdown | verbatim |
| CS231n NN3 | cs231n_neural_networks_3.md | markitdown | verbatim |
| FSDL Spring 2021 L7 | fsdl_spring2021_lecture7.md | markitdown | verbatim |
| Irpan RL hard | alexirpan_rl_hard.md | markitdown | verbatim |
| amid.fish reproducing | amid_fish_reproducing_deep_rl.md | markitdown | verbatim |
| Slavv 37 reasons | slavv_37_reasons_nn.md | markitdown | verbatim |
| CS229 ML advice | cs229_ml_advice.md | markitdown | verbatim |
| McCandlish 2018 | mccandlish_2018_large_batch.md | markitdown | verbatim |
| William Falcon notes | williamfalcon_deeprl_hacks.md | markitdown | verbatim |
| Goodfellow Ch15 | goodfellow_ch15_representation_learning.md | markitdown | verbatim |
| Deep Learning Book | deeplearning_book.md | markitdown | verbatim |
| Reddit RL tips 7s8px9 | reddit_rl_practical_tips_7s8px9.md | markitdown | verbatim |
| Reddit RL debug 9sh77q | reddit_rl_debugging_tips_9sh77q.md | markitdown | verbatim |
| Reddit RL roadblocks | reddit_rl_roadblocks_bzg3l2.md | markitdown | verbatim |
| Reddit Schulman 5hereu | reddit_schulman_nuts_bolts_5hereu.md | markitdown | verbatim |
| Reddit ICML tutorial | reddit_icml2017_tutorial_levine_6vcvu1.md | markitdown | verbatim |
| Reddit DRL bootcamp | reddit_deeprl_bootcamp_2017_75m5vd.md | markitdown | verbatim |

## Quote Verification Notes

- Schulman subtitles contain auto-generated transcription errors (e.g., "mean insanity deviation" should be "mean and standard deviation"). Quotes used verbatim from file; errors are in the source, not introduced by us.
- Schulman PDF (joschu_nuts_and_bolts.md) has markitdown conversion artifacts (`(cid:73)` bullet markers, table formatting). Core text is present but formatting is messy.
- All other evidence files appear to be clean markitdown conversions.
- 15 key quotes were manually spot-checked against evidence files. All matched.
- Quotes from subagent extractions were cross-referenced with direct file reads.

## Blockers / Caveats

- Argdown verifier passes clean: `npx @argdown/cli json` exports 14 arguments, 45 statements, 14 relations. Fixed: 44 blank lines inside PCS blocks, bracket escaping in FSDL quote.
- Some evidence files (especially Schulman PDF) have conversion artifacts that may cause verifier failures on exact quote matching.
- The argdown uses auto-generated YouTube subtitles as a source; these contain transcription errors that are present in the evidence file.

## Coverage Summary

| SKILL.md Claim | Sources Used | Independent Sources |
|---|---|---|
| Normalize inputs mean=0 std=1 | Schulman, FSDL, Slavv | 3 |
| Overfit tiny dataset first | CS231n, FSDL, Goodfellow | 3 |
| Assume you have a bug | Jones, Goodfellow | 2 |
| Seed variance is extreme | Schulman, Henderson, Irpan | 3 |
| Use bigger batch sizes | Schulman (x2), McCandlish | 2 (Schulman slides + talk counted as 1) |
| Hand-scale rewards, don't shift mean | Schulman, Jones, Henderson | 3 |
| Use reference implementations | Jones, Rahtz | 2 |
| Pursue anomalies | Jones, Rahtz | 2 |
| Log everything | Rahtz, Goodfellow | 2 |
| Random HP search | CS231n/Bergstra, Schulman | 2 |

| Probe environments for RL | Jones | 1 (but applies general isolation principle) |
| Policy entropy / KL diagnostics | Schulman | 1 (but built into major frameworks) |

## Claims NOT Covered in Argdown (lower priority or single-source)
- Gradient clipping masks problems (CS231n mentions, but as a technique not a warning)
- Final layer zero init for policy (Schulman only)
- Loss surface analysis / gradient quiver plots (original to SKILL, no external source)
- Sweep methodology with within-group z-scores (original to SKILL)
