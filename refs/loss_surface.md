# Loss surface & gradient analysis (no model required)

Appendix to the [ML Debugging skill](../SKILL.md). A trick worth reaching for when a *loss* (not the whole model) is misbehaving: visualize its surface and gradient flow directly, feeding synthetic tensors into the loss sub-components. No model, forward pass, or GPU, just the math. Five minutes of plotting often saves hours of squinting at training curves.

When you'd look this up: a new or custom loss behaves oddly; a metric is stuck and you suspect the loss shape; you just changed a loss formula and want to confirm gradients still flow at the operating point (not just at init); you're comparing two loss variants and want to see their gradient fields side by side.

## The method

1. Identify each loss sub-component as a function of its immediate inputs.
2. Pick 1-2 axes that matter (the "natural axes" you reason about when you think about the loss).
3. Grid over those axes, feed through the loss, call `.backward()`, collect gradients.
4. Plot: contour heatmap + quiver overlay (negative gradient = the direction the optimizer moves).
5. Build a summary table: component x representative_input -> loss_value, grad_value. Flag zero or non-finite gradients.

```py
# ── 2D loss surface with gradient quiver ──────
def analyze_component(loss_fn, x_range, y_range, n=80):
    xs = torch.linspace(*x_range, n)
    ys = torch.linspace(*y_range, n)
    X, Y = torch.meshgrid(xs, ys, indexing='ij')
    x_flat = X.flatten().requires_grad_(True)
    y_flat = Y.flatten().requires_grad_(True)

    losses = loss_fn(x_flat, y_flat)       # vectorized, returns (n*n,)
    losses.sum().backward()

    loss_grid = losses.detach().reshape(n, n)
    gx = x_flat.grad.reshape(n, n)
    gy = y_flat.grad.reshape(n, n)

    # contourf(X, Y, loss_grid) + quiver(X, Y, -gx, -gy)
    # negative gradient = direction optimizer moves

# ── Gradient flow verification table ──────────
# For each component, evaluate at representative inputs
# (zero, small, converged, degenerate). Report loss + grad.
# Flag: zero grad (dead zone), non-finite (numerical issue).
#
# | Component       | Param   | Input        | Loss     | Grad     |
# |-----------------|---------|--------------|----------|----------|
# | barrier_penalty | v       | v=0.0        | +0.000   | +0.000   |  <-- zero grad!
# | barrier_penalty | v       | v=0.5        | +12.50   | +50.00   |
# | pair_loss       | dot_pos | (0.3, -0.3)  | -2.340   | -3.000   |
# | pair_loss       | dot_neg | (0.3, -0.3)  | -2.340   | +3.000   |  <-- antisym, good
# | pair_loss       | dot_pos | (0.0, 0.0)   | +0.000   | +0.000   |  <-- dead at init!
```

## What to look for

| Pattern | Meaning | Action |
|---------|---------|--------|
| Gradient arrows point toward desired region | Loss is well-shaped | Ship it |
| Large flat region (zero gradient) | Dead zone: optimizer stuck if it lands here | Add curvature, change init, or reparameterize |
| Gradient magnitude 1000x in one axis vs another | Imbalanced: one axis dominates | Rescale, use log-space, or normalize |
| Saddle point at origin | Common with product-form losses (A*B) | Switch to additive (log A + log B) for independent gradients |
| Arrows point away from desired region | Loss is wrong or has an unexpected local min | Rethink the formula |
| Non-finite values in a region | Numerical issue (log(0), 0/0) | Add eps, clamp, or use log1p |

## The log-space decomposition trick

When your loss is a product of factors A*B and one factor can be near zero:

```
# BAD: symlog(A * B), when B~0 the chain rule gives 0 grad to A too
# GOOD: sign * (log|A| + log|B|) gives independent gradients
#   d/dA = 1/A  regardless of B
#   d/dB = 1/B  regardless of A
```

General principle: if you want gradient to flow independently through two factors, decompose multiplicatively in log space.

You can also design surrogate losses that are better behaved but move in the right direction in a better behaved well.
