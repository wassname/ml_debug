# 6.2 Diagnostic code snippets

Copy-paste these. Each tests one thing.

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

**Overfit-one-batch test**
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
