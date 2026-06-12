# 6.2 Diagnostic code snippets

Part of the [ML Debugging skill](../SKILL.md), section 6.2.

Here are various idea's on how to cheaply diagnose parts of your ML pipeline.

**Data pipeline sanity check**
```python
batch = next(iter(train_loader))
for k, v in (batch.items() if isinstance(batch, dict) else enumerate(batch)):
    if isinstance(v, torch.Tensor):
        print(f"{k}: shape={v.shape}, dtype={v.dtype}, "
              f"range=[{v.min():.3f}, {v.max():.3f}], "
              f"mean={v.float().mean():.3f}, std={v.float().std():.3f}, "
              f"nan={v.isnan().sum()}, inf={v.isinf().sum()}")
    else:
        print(f"{k}: type={type(v)}, len={len(v) if hasattr(v, '__len__') else 'scalar'}")
# Check: inputs ~mean 0, std 1? Labels in expected range? No NaN/Inf? Shapes match model?
```

**Init loss check**
```python
model.eval()
with torch.no_grad():
    batch = next(iter(train_loader))
    out = model(batch['input'])  # adapt to your interface
    loss = loss_fn(out, batch['target'])
    print(f"Init loss: {loss.item():.4f}")

# Expected init loss (random predictions):
# - CrossEntropy, C classes:  -ln(1/C) = ln(C)
#     C=2: 0.693, C=10: 2.303, C=100: 4.605, C=1000: 6.908
# - Binary CrossEntropy:      -ln(0.5) = 0.693
# - MSE (targets ~N(0,1)):    ~1.0 (if init outputs ~0) or ~var(targets)
# - L1 (targets ~N(0,1)):     ~0.8
#
# If init loss << expected: model is cheating (data leakage, shortcut)
# If init loss >> expected: wrong loss fn, bad init, or data pipeline broken
```

**Overfit-one-batch test** [Ng / torch lightning]
```python
model.train()
batch = next(iter(train_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    optimizer.zero_grad()
    out = model(batch['input'])
    loss = loss_fn(out, batch['target'])
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss.item():.4f}  grad_norm={grad_norm:.4f}")

# Expected: loss drops to ~0 within 200 steps.
# If not: model can't even memorize 1 batch -- architecture or gradient problem.
```

**Gradient flow check (per-layer)**
```python
loss.backward()
for name, p in model.named_parameters():
    if p.grad is not None:
        g = p.grad
        print(f"{name:40s}  grad: mean={g.mean():+.2e}, std={g.std():.2e}, "
              f"max={g.abs().max():.2e}, zero%={100*(g==0).float().mean():.0f}")
    else:
        print(f"{name:40s}  grad: None")  # <-- not in computation graph!
# Check: no None grads (disconnected), no all-zero grads (dead layer),
# no huge grads (explosion), reasonable magnitude across layers.
```

**NaN/Inf detector hooks**
```python
def nan_hook(module, input, output):
    def _check(t, label):
        if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
            raise RuntimeError(
                f"NaN/Inf in {module.__class__.__name__} {label}, "
                f"shape={t.shape}, nan={t.isnan().sum()}, inf={t.isinf().sum()}")
    if isinstance(output, torch.Tensor):
        _check(output, "output")
    elif isinstance(output, dict):
        for k, v in output.items():
            _check(v, f"output[{k!r}]")
    elif isinstance(output, (tuple, list)):
        for i, o in enumerate(output):
            _check(o, f"output[{i}]")

for name, module in model.named_modules():
    module.register_forward_hook(nan_hook)
# Run one forward pass. First module to raise = source of the NaN.
```

**Random input test** [Slavv]
```python
# Pass random noise instead of real data. If loss/error behaves the same,
# the data pipeline is destroying information before the model sees it.
model.eval()
real_batch = next(iter(train_loader))
fake_input = torch.randn_like(real_batch['input'])
with torch.no_grad():
    real_out = model(real_batch['input'])
    fake_out = model(fake_input)
    real_loss = loss_fn(real_out, real_batch['target']).item()
    fake_loss = loss_fn(fake_out, real_batch['target']).item()
    print(f"Real input loss: {real_loss:.4f}")
    print(f"Random input loss: {fake_loss:.4f}")
# If similar: model isn't using the input. Check preprocessing, data loading, feature selection.
# If very different: model sees real signal. Problem is elsewhere.
```

**NaN poisoning (leakage tracer)** [Wassname
```python
# Leakage can hide anywhere: normalization fit on the full dataset, target
# leaking into features, window functions peeking ahead, bad splits. Instead
# of auditing each spot, inject NaN where information must NOT come from
# (the future, the test set, the label) and run the real pipeline. NaN is
# absorbing under +,-,*,/ so it spreads like dye: if any "past"/train output
# is NaN, you have a leak, and you can bisect the pipeline to find the stage
# where it crossed.
import numpy as np
X = np.random.randn(1000, n_features)
y = np.random.randn(1000)
X[cutoff:] = np.nan          # poison the future / test rows
y[cutoff:] = np.nan

Xt, yt = pipeline(X, y)       # the REAL pipeline: features, scaling, splits, windowing
assert np.isfinite(Xt[:cutoff]).all(), "leak: future reached past features"
assert np.isfinite(yt[:cutoff]).all(), "leak: future reached past targets"
# To localize: assert finiteness after each pipeline stage; first failing
# stage is where the leak crosses.

# CAVEAT false negatives (dye silently filtered -- false assurance):
#   pandas mean/std/sum default to skipna=True; np.nanmean; dropna/fillna;
#   imputers; df.rolling(...).mean() skips NaN too.
#   Fallback: poison with a huge sentinel (1e12) instead -- survives nanmean
#   and shows up as an absurd value in anything it touches.
# CAVEAT false positives (dye spreads along a legitimate axis):
#   softmax over an axis containing NaN goes all-NaN even with a CORRECT
#   additive -inf causal mask (NaN + -inf = NaN). So this cannot validate
#   causal masking inside a transformer -- use the gradient check below.
#   But NaN crossing via batch statistics is often a TRUE positive: a scaler
#   fit on train+test lets test rows poison train features. That's the leak.
```

**Backprop-to-input dependency check** [Karpathy 2019]
```python
# The gradient-based dual of NaN poisoning: works INSIDE models where NaN
# gives false positives (attention softmax, batch/layer stats).
# Karpathy: "set the loss to be something trivial like the sum of all outputs
# of example i... ensure that you get a non-zero gradient only on the i-th input."
# Catches view-instead-of-transpose bugs that mix info across the batch dim.

# Batch independence: output i must depend only on input i
x = torch.randn(8, seq, dim, requires_grad=True)
model(x)[3].sum().backward()
assert (x.grad[[0,1,2,4,5,6,7]] == 0).all(), "leak across batch dim"

# Causal masking: output at t must not depend on inputs > t
x = torch.randn(1, seq, dim, requires_grad=True)
t = seq // 2
model(x)[0, t].sum().backward()
assert (x.grad[0, t+1:] == 0).all(), "leak: position t sees the future"
# Run in eval mode; dropout and exotic attn kernels can add noise.
```

**Prime dimension trick** [Slavv]
```python
# Use prime/weird numbers for each dimension to catch silent broadcasting.
# If batch=7, seq=13, hidden=17, any mismatched reshape/view that "works"
# by accident with powers-of-2 will fail with primes.
x = torch.randn(7, 13, 17)  # (batch=7, seq=13, hidden=17)
out = model(x)
print(f"in={x.shape} -> out={out.shape}")
# If this crashes but normal shapes don't: you have a broadcasting bug.
```

**Class imbalance check**
```python
from collections import Counter
all_labels = []
for batch in train_loader:
    labels = batch['target'] if isinstance(batch, dict) else batch[1]
    all_labels.extend(labels.flatten().tolist())
counts = Counter(all_labels)
total = sum(counts.values())
for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  class {cls}: {n:6d} ({100*n/total:.1f}%)")
# Ratio > 10:1 = likely need weighted loss or resampling.
# Ratio > 100:1 = model will predict majority class and look "accurate".
```

**Confidence-sorted error inspection** [common practice, cf. FSDL error analysis]
```python
# Find the model's most confident wrong predictions. These reveal
# systematic bugs (e.g., cropping cutting off relevant features).
model.eval()
errors = []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch['input'])
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        wrong = predicted != batch['target']
        for i in wrong.nonzero(as_tuple=True)[0]:
            errors.append((confidence[i].item(), predicted[i].item(),
                           batch['target'][i].item(), i.item()))
errors.sort(reverse=True)  # most confident mistakes first
for conf, pred, true, idx in errors[:10]:
    print(f"  conf={conf:.3f} predicted={pred} true={true} idx={idx}")
# Inspect the actual inputs for these indices. Pattern = systematic bug.
```

**Update-to-data ratio check** [Karpathy nn-zero-to-hero Lec 4; evidence: karpathy_nn_zero_to_hero_lec4_diagnostics.md]
```python
# Track during training: how large are updates relative to parameter magnitudes?
# Target: ~1e-3 (log10 ~ -3). Much higher = LR too large. Much lower = LR too small.
ud = []
# Inside training loop (after optimizer.step()):
with torch.no_grad():
    ud.append({
        name: ((lr * p.grad).std() / p.data.std()).log10().item()
        for name, p in model.named_parameters()
        if p.grad is not None and p.ndim >= 2
    })
# After training, plot per-layer ratios:
import matplotlib.pyplot as plt
for name in ud[0]:
    plt.plot([d[name] for d in ud], label=name)
plt.axhline(-3, color='k', linestyle='--')  # target ratio
plt.legend(); plt.ylabel('log10(update/param ratio)'); plt.show()
# If a layer's ratio is much above -3: reduce LR or add gradient clipping.
# If much below -3: that layer is barely updating -- possible dead/frozen layer.
```

**Weight/bias distribution check** [Slavv, CS231n]
```python
for name, p in model.named_parameters():
    print(f"{name:40s}  mean={p.data.mean():+.4f}  std={p.data.std():.4f}  "
          f"min={p.data.min():+.4f}  max={p.data.max():+.4f}  "
          f"shape={list(p.shape)}")
# Healthy: roughly Gaussian, std ~0.01-1.0 depending on init scheme.
# Bad signs: all zeros, huge values (>100), std ~0 (collapsed), NaN.
# After training: weights diverging to +/-inf = exploding. All same value = dead.
```

---

## JAX diagnostic equivalents

| Diagnostic | PyTorch | JAX |
|------------|---------|-----|
| NaN detection | `torch.autograd.detect_anomaly()` | `jax.config.update("jax_debug_nans", True)` |
| Gradient check | `torch.autograd.gradcheck(fn, inputs)` | `jax.test_util.check_grads(fn, args, order=2)` |
| Eager debug (no compile) | N/A (already eager) | `jax.config.update("jax_disable_jit", True)` |
| Print inside compiled | N/A | `jax.debug.print("{x}", x=x)` |
| Breakpoint inside compiled | `pdb.set_trace()` | `jax.debug.breakpoint()` |
| Runtime assertions inside compiled | `assert` | `jax.experimental.checkify` |
