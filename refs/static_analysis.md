# 6.1 Static analysis: grep for silent bugs

Run these searches on the codebase before anything else. Each catches a common bug that produces no error but wrong results.

**Shape mismatches (silent broadcasting)**
```
# Grep patterns:
\.view\(|\.reshape\(            # check dims match intent
unsqueeze\(|squeeze\(           # dimension insertion/removal
\.expand\(|\.repeat\(           # broadcasting
# Action: for every hit, trace the tensor shape backward. Add assert statements.
```

**Autograd breakers**
```
# Grep patterns:
\.detach\(\)                    # breaks gradient flow
\.data\b                        # bypasses autograd entirely
with torch\.no_grad             # check this isn't wrapping training code
\.item\(\)                      # in a loss computation = broken
\.numpy\(\)                     # in forward pass = broken
# Action: every .detach() should have a comment explaining WHY grad is intentionally stopped.
```

**Missing train/eval mode**
```
# Grep patterns:
\.train\(\)                     # count occurrences
\.eval\(\)                      # should pair with .train()
# Action: verify .eval() before every val loop, .train() before every train loop.
# Dropout and batchnorm behave differently -- this silently degrades results.
```

**In-place ops on tensors requiring grad**
```
# Grep patterns:
\+=|\-=|\*=|/=                  # in-place assignment on tensors
\.add_\(|\.mul_\(|\.zero_\(     # in-place methods
\[.*\]\s*=[^=]                  # index assignment (excludes ==)
# Action: in-place ops on leaf tensors with requires_grad=True corrupt autograd.
# Replace x += y with x = x + y.
```

**Double softmax (softmax input to CrossEntropyLoss)**
```
# Grep patterns:
CrossEntropyLoss|cross_entropy  # expects raw logits
softmax|log_softmax|\.softmax   # if applied BEFORE CrossEntropyLoss = double softmax
# Action: CrossEntropyLoss = log_softmax + NLLLoss internally.
# If you softmax first, CE computes log_softmax(softmax(x)) -- the softmax
# compresses logits into (0,1), so log_softmax sees near-uniform inputs.
# Gradients vanish. Loss plateaus near ln(n_classes).
```

**Wrong optimizer step ordering**
```
# Grep patterns -- verify this exact order exists:
# 1. optimizer.zero_grad()
# 2. loss.backward()
# 3. [optional: clip_grad_norm_]
# 4. optimizer.step()
# 5. [optional: scheduler.step()]
# Common bugs: zero_grad after backward (kills grads), step before backward (stale grads),
# scheduler.step() in wrong loop: per-epoch schedulers (StepLR, CosineAnnealingLR)
# called per-batch = decays too fast. Per-step schedulers (OneCycleLR) called per-epoch = too slow.
```

**Broadcasting traps**
```python
# Diagnostic: print shapes at every binary operation between tensors of different ndim
# Shapes (3,) and (3,1) silently broadcast to (3,3) -- probably not intended.
# Shapes (B,1) and (B,N) broadcast fine but verify it's intentional.
a = torch.randn(3)
b = torch.randn(3, 1)
print((a + b).shape)  # (3, 3) -- wanted (3,)?
```

**Wrong loss sign**
```
# Grep patterns:
maximize|ascent              # gradient ascent when descent intended?
\-\s*loss                    # negating loss -- intentional (e.g., reward maximization)?
1\.0\s*-\s*|1\s*-\s*         # 1 - metric as loss -- is the metric bounded [0,1]?
# Action: verify that minimizing the loss = improving the metric you care about.
```

**Frozen parameters not intended**
```
# Grep patterns:
requires_grad\s*=\s*False    # intentional freeze?
\.freeze\(|\.requires_grad_  # parameter freezing
for.*param.*\.parameters     # check nothing is skipped
# Diagnostic:
for name, p in model.named_parameters():
    if not p.requires_grad:
        print(f"FROZEN: {name}")
```

**Data leakage**
```
# Grep patterns:
\.fit_transform\(             # on test data = leakage
train_test_split.*shuffle=True  # for time series = leakage
# Action: fit on train only, transform on both. Use temporal split for time series.
```

**Class imbalance**
```
# Grep patterns:
CrossEntropyLoss\(\)          # no weight= argument? check if classes balanced
weight=.*class                # existing balancing -- verify weights are correct
# Diagnostic: count labels per class (see diagnostics.md "Class imbalance check").
# 100:1 ratio with unweighted loss = model predicts majority class.
```
