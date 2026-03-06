# Heat Exchanger PINN Notes

Project-specific notes for heat exchanger PINNs with phase change. Extracted from the general [PINN skill](../SKILL.md) to keep it domain-agnostic.

---

## Complexity ladder (heat exchanger example)

| Level | Model | Assumptions | What it tests |
|---|---|---|---|
| 0 | Persistence: y(t) = y(t-1) | None | Does model capture any dynamics? |
| 1 | Exponential decay: T(t) = T_ss + (T_0 - T_ss) exp(-t/tau) | First-order response | Dominant thermal mode |
| 2 | Linear state-space / DMDc: fit A, B via OLS on finite differences (or Dynamic Mode Decomposition with control) | Linear dynamics | Near-linear regime |
| 3 | Pure data MLP (no physics loss) | NN can interpolate | If PINN doesn't beat this, physics constraint hurts |
| 4 | Effectiveness-NTU closed form | 1D, constant Cp, steady state | Analytical sanity check |
| 5 | 1D ODE with constant Cp (scipy solve_bvp) | 1D, constant Cp | Classical numerical baseline |
| 6 | 1D ODE with enthalpy (scipy solve_bvp) | 1D, variable properties | Phase change handling |
| 7 | 1D PINN with enthalpy | 1D, learned U | Differentiable model, parameter estimation |
| 8 | 2D PINN with radial maldistribution | 2D, learned U + u_r | Full physics |

---

## Hard BC example (heat exchanger)

For our heat exchanger: u_{r,tube}(z,r) = r(R-r) * NN(z,r) enforces zero radial velocity at r=0 and r=R by construction. Same principle as Sukumar & Srivastava 2022.

---

## Known failure modes

**Degenerate U -> 0**: When learning a heat transfer coefficient U jointly with the network, the optimizer may find U -> 0 (no heat transfer, flat profiles). This satisfies the PDE trivially.
- Fix: bound U in [100, 5000], initialize near a physically reasonable value (~500).
- Fix: penalize |dT/dz| being too small (the network should predict temperature change).

**Sign errors in counterflow**: The counterflow BVP in z-coordinates has both fluids' enthalpies decreasing in z:
  - dh_t/dz = -U*A'*(T_t - T_s)/m_dot_t  (tube cools, < 0 since T_t > T_s)
  - dh_s/dz = -U*A'*(T_t - T_s)/m_dot_s  (same sign! shell also decreases in z, but its flow is -z so it warms)
  The counterflow is encoded in BCs (h_t at z=0, h_s at z=L), not in opposing ODE signs.
  Common mistake: putting dh_s/dz = +q gives dh_s/dz > 0 (shell hot at z=0 exit, cold at z=L entry -- wrong).

**NN vs scalar parameter**: When a scalar (U) and a NN (temperature field) are optimized jointly, the NN is easier to update (more parameters, more gradient signal). The scalar gets stuck. Fix: use different learning rates, or estimate U separately first.

---

## Property Mappings (EoS for PINNs)

For heat exchangers with phase change, the T(h) mapping from REFPROP/GERG-2008 must be made differentiable.

**Recipe:**
1. Generate REFPROP data: T vs h at constant P, dense enough near the phase boundary (~200 points over T = [100, 300] K).
2. Fit a cubic spline to get ~20 minimal knots (reject redundant knots where the curve is nearly linear).
3. Build PCHIP on those knots for monotonicity guarantee (cubic spline can wiggle; PCHIP cannot).
4. Wrap for PyTorch autograd:
   ```python
   class PchipFunction(torch.autograd.Function):
       @staticmethod
       def forward(ctx, h):
           T = pchip_interp(h.detach().numpy())  # scipy
           ctx.save_for_backward(h)
           return torch.tensor(T, dtype=h.dtype, device=h.device)

       @staticmethod
       def backward(ctx, grad_output):
           h, = ctx.saved_tensors
           dTdh = pchip_interp.derivative()(h.detach().cpu().numpy())  # scipy
           return grad_output * torch.tensor(dTdh, dtype=h.dtype, device=h.device)
   ```
5. Use float64 throughout. float32 loses precision near the phase boundary where dT/dh is small.

**Why h -> T and not T -> h?** The h(T) direction has a steep region near phase change where dh/dT = Cp -> infinity. The inverse T(h) is smooth and monotone everywhere. Always solve in h space, convert to T only for losses and visualization.

---

## Initial Conditions and Multi-Episode Training

IC is a loss term, but in practice ICs from real plant data are noisy and uncertain.

**Options (in order of complexity):**
1. Hard-code IC from first observed data point. Simplest; biased if sensor lags.
2. Learnable IC: optimize IC scalars jointly with the network. Adds ~n_states parameters. Works well with L-BFGS polish.
3. Soft IC loss: penalize `||u(t=0) - u_obs(0)||^2` as one loss term. Lets optimizer trade off IC vs physics fit.

**Multi-episode training** (e.g., ~10 cold restarts over a year):
- Each episode needs its own IC.
- Common pattern: learn per-episode IC vector `ic_k` while sharing the physics network across episodes.
- Warm-start trick: if you have a pure-data pre-fit (Level 3 baseline), initialize PINN data-loss weights from that fit.

---

## Domain decomposition (heat exchanger note)

Could split axially (cold end / hot end) if the solution has different character in different regions.

---

## Heat-exchanger-specific symptom table entries

| Symptom | Likely cause |
|---|---|
| Loss decreases then explodes | Numerical instability in T(h) near phase boundary |
| Loss NaN | temperature outside EoS range |
| U converges to 0 or bound | Degenerate solution, check sign convention |
| PINN worse than pure-data MLP | Physics constraint fighting the data |
