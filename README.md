# wassname's ML Debugging Folklore

In an attempt to upskill the machine learning debugging on AI coding assistants (and humans), I've collected high quality sources on how to debug machine learning projects, focusing on the mindset and the "taste". When I started ML I went searching for discussions on best practices, and started a few discussions of my own and they helped me a lot, over the years I've collected good ones. I hope they can help others, as well as help in auto research setups. This intro is human written, and the below is AI written with human guidance.

## Use as a Claude skill

```
/skills add https://github.com/wassname/ml_debug
```

Or paste `SKILL.md` into your system prompt / context when debugging.

## What's here

- **[SKILL.md](SKILL.md)** -- the main artifact. Load into an LLM agent's context as a debugging skill. A short calibration note, then the folklore itself: verbatim sourced quotes from practitioners, general lessons first, modern transformers and LLM fine-tuning in their own section.

- **[PLAYBOOK.md](PLAYBOOK.md)** -- the synthesized long-form: mental models, practitioner priors, step catalogs, symptom tables, the agent debugging loop, triage, and anti-patterns. Menus of hypotheses distilled from the same sources, not quotes. Deeper one-off tricks (loss-surface analysis, stuck-metric diagnosis, sweep reliability) live in [refs/](refs/).

- **[docs/evidence/](docs/evidence/)** -- frozen local copies of source material (blog posts, talks, papers, reddit threads). Claims in SKILL.md link back to exact quotes here.

## Citation

```bibtex
@misc{wassname2026mldebug,
  title = {ML Debugging Folklore: A Practitioner Debugging Skill for LLM Agents},
  author = {Michael J. Clark},
  year = {2026},
  url = {https://github.com/wassname/ml_debug/}
}
```
