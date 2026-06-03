# wassname's ML Debugging Folklore

In an attempt to upskill the ML debugging on AI coding assistants (and humans), I've collected high quality sources on ML debugging and the mindset and the "taste". When I started ML I went searching for discussions on best practices, and started a few discussions of my own and they helped me a lot, I hope they can help others. This intro is human written, and the below is AI written with human guidance.

## Use as a Claude skill

```
/skills add https://github.com/wassname/ml_debug
```

Or paste `SKILL.md` into your system prompt / context when debugging.

## What's here

- **[SKILL.md](SKILL.md)** -- the main artifact. Load into an LLM agent's context as a debugging skill. Leads with the mindset (calibrate, mental models, general debugging tricks, and reading a working implementation when stuck), then a folklore section of sourced quotes, then an LLM-agent playbook (debugging loop, triage menu, anti-patterns). Deeper one-off tricks (loss-surface analysis, stuck-metric diagnosis, sweep reliability) live in [refs/](refs/).

- **[docs/evidence/](docs/evidence/)** -- frozen local copies of source material (blog posts, talks, papers, reddit threads). Claims in SKILL.md link back to exact quotes here.
