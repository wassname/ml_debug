# ML Debugging Folklore

Deep research to uplift LLMs for ML debugging. Opinionated by source selection.

Distilled from Schulman's "Nuts and Bolts" talk, Andy Jones' debugging guide, Goodfellow Ch11, CS231n, FSDL, and more. Every non-obvious claim is traced to a verbatim source quote in [`docs/ml_debug_folklore.argdown`](docs/ml_debug_folklore.argdown) (vargdown format).

**Author**: [wassname](https://github.com/wassname)

## What's here

- **[SKILL.md](SKILL.md)** -- the main artifact. Designed to be loaded into an LLM agent's context as a debugging skill. Parts 1-5 are reference knowledge; Part 6 is a runnable triage protocol (grep patterns, diagnostic code snippets, decision tree); Part 7 is debugging mental models and practitioner priors.

- **[docs/ml_debug_folklore.argdown](docs/ml_debug_folklore.argdown)** -- vargdown source map. Traces each claim to an exact quote + file in `docs/evidence/`.

- **[docs/evidence/](docs/evidence/)** -- frozen local copies of source material (blog posts, talks, papers, reddit threads).

## Use as a Claude skill

```
/skills add https://github.com/wassname/ml_debug
```

Or paste `SKILL.md` into your system prompt / context when debugging.

## Sources

Schulman (2017), Jones (2021), Rahtz (2018), Goodfellow et al. (Deep Learning book), Karpathy (CS231n), Ng (CS229), FSDL, Henderson et al. (2018), McCandlish et al. (2018), Irpan (2018), Slavv (2017), and Reddit.
