# Spinning Up as a Deep RL Researcher — Joshua Achiam (OpenAI, 2018-10-13)

Source: https://spinningup.openai.com/en/latest/spinningup/spinningup.html . Verbatim excerpts (the debugging/rigour passages) cached for the ML-debugging skill.

---

## Learn by Doing

**Simplicity is critical.** You should organize your efforts so that you implement the simplest algorithms first, and only gradually introduce complexity. If you start off trying to build something with too many moving parts, odds are good that it will break and you'll lose weeks trying to debug it.

**Focus on understanding.** Writing working RL code requires clear, detail-oriented understanding of the algorithms. This is because **broken RL code almost always fails silently,** where the code appears to run fine except that the agent never learns how to solve the task. Usually the problem is that something is being calculated with the wrong equation, or on the wrong distribution, or data is being piped into the wrong place. Sometimes the only way to find these bugs is to read the code with a critical eye, know exactly what it should be doing, and find where it deviates from the correct behavior.

**But don't overfit to paper details.** Sometimes, the paper prescribes the use of more tricks than are strictly necessary, so be a bit wary of this, and try out simplifications where possible. For example, the original DDPG paper suggests a complex neural network architecture and initialization scheme, as well as batch normalization. These aren't strictly necessary, and some of the best-reported results for DDPG use simpler networks. As another example, the original A3C paper uses asynchronous updates from the various actor-learners, but it turns out that synchronous updates work about as well.

**Don't overfit to existing implementations either.** Study existing implementations for inspiration, but be careful not to overfit to the engineering details of those implementations. RL libraries frequently make choices for abstraction that are good for code reuse between algorithms, but which are unnecessary if you're only writing a single algorithm or supporting a single use case.

**Iterate fast in simple environments.** To debug your implementations, try them with simple environments where learning should happen quickly [...]. Don't try to run an algorithm in Atari or a complex Humanoid environment if you haven't first verified that it works on the simplest possible toy task. Your ideal experiment turnaround-time at the debug stage is <5 minutes (on your local machine) or slightly longer but not much.

**If it doesn't work, assume there's a bug.** Spend a lot of effort searching for bugs before you resort to tweaking hyperparameters: usually it's a bug. Bad hyperparameters can significantly degrade RL performance, but if you're using hyperparameters similar to the ones in papers and standard implementations, those will probably not be the issue. Also worth keeping in mind: sometimes things will work in one environment even when you have a breaking bug, so make sure to test in more than one environment once your results look promising.

**Measure everything.** Do a lot of instrumenting to see what's going on under-the-hood. The more stats about the learning process you read out at each iteration, the easier it is to debug—after all, you can't tell it's broken if you can't see that it's breaking. I personally like to look at the mean/std/min/max for cumulative rewards, episode lengths, and value function estimates, along with the losses for the objectives, and the details of any exploration parameters [...]. Also, watch videos of your agent's performance every now and then; this will give you some insights you wouldn't get otherwise.

## Doing Rigorous Research in RL

**Set up fair comparisons.** If you implement your baseline from scratch [...] it's important to spend as much time tuning your baseline as you spend tuning your own algorithm. This will make sure that comparisons are fair. Also, do your best to hold "all else equal" [...]. Under no circumstances handicap the baseline!

**Remove stochasticity as a confounder.** Beware of random seeds making things look stronger or weaker than they really are, so run everything for many random seeds (at least 3, but if you want to be thorough, do 10 or more). [...] There's potentially enough variance that two different groups of random seeds can yield learning curves with differences so significant that they look like they don't come from the same distribution at all.

**Run high-integrity experiments.** Don't just take the results from the best or most interesting runs to use in your paper. Instead, launch new, final experiments [...] and precommit to report on whatever comes out of that. This is to enforce a weak form of preregistration: you use the tuning stage to come up with your hypotheses, and you use the final runs to come up with your conclusions.

**Check each claim separately.** [...] run an ablation analysis. Any method you propose is likely to have several key design decisions [...] By systematically evaluating what would happen if you were to swap them out with alternate design choices, or remove them entirely, you can figure out how to correctly attribute credit for the benefits your method confers.
