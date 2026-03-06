---
name: pinn-debug
description: "PINN (Physics-Informed Neural Network) training best practices and debugging. Use when building, debugging, or optimizing PINNs for PDEs, ODEs, or physics-constrained learning problems. Sub-skill of ml-debugging."
---


# PINN Training Best Practices

https://gist.github.com/wassname/a68314f731002b32aa637c3d43a081c9/edit


Consolidated from: NeuralPDE.jl tests/docs, ConFIG repo, Wang et al. 2021, Rathore et al. 2024 (ICML), ml_debug folklore, and practical experience with heat exchanger PINNs.

Epistemic status: Patterns confirmed across multiple sources. Where sources disagree, noted. Paper claims marked with credence estimates.

---

## 0. Before You PINN: Complexity Ladder and Baselines

PINNs are complex. Before trusting a PINN, work up the complexity ladder and compare against simpler baselines at each level. Each level should beat the previous one or you're adding complexity for nothing.

**Make complexity pay rent.** If a fancier model doesn't improve on the simpler one, the added physics/architecture is either wrong, badly scaled, or unnecessary.

### Complexity ladder (heat exchanger example)

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

At each level, brainstorm:
- What assumption am I adding/relaxing?
- What does this buy me (lower RMSE, new physics captured)?
- What breaks if I simplify further?
- **Which inputs does the model need to be conditional on?** If you want what-if scenarios ("what happens if I double the flow rate?"), the model must take that variable as an explicit input -- not bake it in as a constant. A model trained at fixed m_dot cannot generalize to m_dot * 2, even in-distribution. Decide the conditioning variables before building the model.

**Try the simplest baselines first.** If scipy.optimize.minimize + solve_bvp fits your data to 2 K RMSE, you probably don't need a PINN.

**Steady-state before dynamic.** If the problem has a transient component, solve the steady-state first. It's strictly simpler (no IC, no temporal causality), gives a reference solution, and the PINN for transient can be initialized from the steady-state solution. Many process systems spend >90% of operating time near steady-state anyway.

Report RMSE, MAE, and trajectory plots for each baseline alongside the PINN. Physics-informed models should win on extrapolation and long-horizon rollout even if in-sample fit is similar.

**Train/test split**: Use temporal split (not random). First 75% train, last 25% test. Random splitting leaks temporal correlation.

---

## 1. Nondimensionalization

Nondimensionalize the governing equations *before* feeding them to the network. Raw SI units (T in Kelvin, h in J/kg, z in meters) create loss terms with wildly different magnitudes -- this is exactly the multi-scale problem that adaptive weighting tries to fix downstream. Nondimensionalizing fixes it at the source.

**Recipe:**
1. Pick characteristic scales: T_ref, L_ref, h_ref = Cp_ref * T_ref, etc.
2. Define dimensionless variables: z* = z/L, T* = (T - T_cold)/(T_hot - T_cold), etc.
3. Substitute into the PDE, cancel dimensions.
4. The resulting coefficients are O(1) dimensionless groups (NTU, Biot, etc.).

This also makes the input domain ~[0,1] naturally, which is what NeuralPDE.jl examples assume but never say why.

If you can't nondimensionalize cleanly (unknown material properties, mixed units), at minimum z-score each input/output channel so the network sees zero-mean unit-variance data.

> Rathore et al. 2024: "The condition number grows polynomially with nres" -- but this is in raw units. Nondimensionalization reduces the effective condition number by making all PDE coefficients O(1).
> Source: https://arxiv.org/abs/2402.01868, Section 5, Theorem 8.4

---

## 2. Architecture

From NeuralPDE.jl tests/docs + Wang et al. 2021:

- **Depth**: 2-3 layers most common; 4 sometimes; 5 is rare. Deeper doesn't help and hurts conditioning.
- **Width**: 16-64 (start at 32). Occasionally 128 for harder problems.
- **Activations**: tanh dominates. SiLU/GeLU sometimes. sigmoid for some ODEs. Final layer linear.
- **Multi-output**: One chain per dependent variable: `[Chain(Dense(in, n, act), Dense(n, n, act), Dense(n, 1)) for _ in 1:k]` or a single shared trunk with separate heads.
- **No**: Fourier features (unless spectral bias is a problem), batch norm, dropout, skip connections, attention. These are for data-rich regimes; PINNs are data-poor + physics-rich.
- **Precision**: float64 for numerical stability. Convert model: `model.double()` or `|> f64` in Julia.
- **Init**: Glorot uniform (Xavier), zero biases. Standard.

**Modified MLP** (Wang et al. 2021, credence ~70%):
> Wang et al. propose a modified MLP with multiplicative interactions: `σ(Wz + b) * U + (1 - σ(Wz + b)) * V` where U, V are linear projections of the input. Authors claim this reduces Hessian stiffness. "Greatly enhances predictive accuracy" for nonlinear PDEs.
> Source: https://arxiv.org/abs/2001.04536, Section 2.6
> Evidence: 49x improvement on Helmholtz, 64x on Klein-Gordon. But only tested by the proposing authors; no independent replication found.

**Random Weight Factorization (RWF)** (arXiv 2210.01274, credence ~60%):
> Factorize each neuron's weight vector as w = s * w_unit, where s is a trainable scalar and w_unit is the unit-normalized direction. This changes the optimization geometry so the loss surface has better-conditioned local minima. "Predictions obtained by RWF are in excellent agreement with ground truth, while other weight parameterizations result in poor or non-physical approximations."
> Source: https://arxiv.org/abs/2210.01274
> Used in the PirateNet architecture alongside causal training, sequence-to-sequence, and Fourier features. Simple to implement as a custom parameterization on Linear layers.
> Credence: plausible mechanism, but proposing-author result; check jaxpi repo for independent adoption.

**PirateNet** (jaxpi library, credence ~55%): Bundles RWF + causal time-marching + seq2seq + Fourier features into one architecture. Good reference implementation when you want all the tricks.
> Source: https://github.com/PredictiveIntelligenceLab/jaxpi

**Symmetry-enforcing architectures** (Julia Ling et al., credence ~75%):
> Instead of data-augmenting with transformed copies, bake symmetries directly into the architecture so every model in the function space is automatically invariant/equivariant. For turbulence closure (Reynolds stress from velocity gradients), custom tensor layers enforce Galilean invariance by construction. "The Galilean invariant model is more accurate than the other models" and generalizes better across flow configurations.
> Source: Ling et al. 2016, "Machine learning strategies for systems with invariance properties." J. Comput. Phys. https://doi.org/10.1016/j.jcp.2016.05.003
> Lecture: Brunton, S. "AI/ML+Physics Part 3 - Designing an Architecture." https://www.youtube.com/watch?v=fiX8c-4K0-Q
> Key distinction: invariance (output unchanged by transformation, e.g., energy is frame-invariant) vs equivariance (output transforms same way as input, e.g., stress tensor rotates with frame). Equivariant architectures are more general. If your PDE has known symmetries (translation, rotation, scaling), enforce them architecturally rather than hoping the optimizer discovers them.
> **Caveat**: This works best for local closure terms (Reynolds stress, turbulence models) and unbounded/periodic domains where the global symmetry holds everywhere. If your domain has boundary conditions that break the symmetry (e.g., a wall breaks rotational invariance), enforcing the symmetry globally in the architecture will prevent the solution from satisfying the BCs -- the architecture will be fighting the problem. In bounded domains, use symmetry-enforcing architectures only for terms where the symmetry genuinely holds (e.g., the constitutive relation), not for the full solution field. Libraries like `e3nn` implement this but add significant computational overhead.

---

## 3. Optimization

The loss landscape of PINNs is ill-conditioned. First-order methods (Adam) converge slowly near the solution. Second-order methods (L-BFGS) can get stuck on strong Wolfe line search. The solution: combine them.

### Recommended workflow

```
1. Adam (lr grid search: {1e-1, 1e-2, 1e-3}) for 1k-11k iterations
   - Escapes saddle points, explores broadly
   - Tolerates noisy gradients
2. L-BFGS (lr=1.0, memory=100) until convergence stalls
   - Preconditions the Hessian, reduces condition number by ~1000x
   - Often stalls: cannot find step size satisfying strong Wolfe conditions
3. (Optional) NysNewton-CG (NNCG) for final polish
   - 1.4-4.3x further improvement in L2RE
   - But 5-300x slower per iteration than L-BFGS
```

> Rathore et al. 2024 (ICML, credence ~80%): "Adam+L-BFGS attains 14.2x smaller L2RE than Adam on convection and 6.07x smaller than L-BFGS on wave." Tested on 3 PDEs (convection, reaction, wave), 5 seeds, widths 50-400.
> Source: https://arxiv.org/abs/2402.01868, Table 1
> Code: https://github.com/pratikrathore8/opt_for_pinns

**Alternative from NeuralPDE.jl**: Stepwise LR decrease with warm-start:
```
Adam(0.1) -> Adam(0.01) -> Adam(0.001)
# Remake problem with u0 = res.u at each stage
```

### Key findings on loss landscape

**Near-zero loss required** (Rathore et al., credence ~85%):
> "A loss of 1e-3 yields L2RE ~ 1e-1, but decreasing loss by 100x to 1e-5 yields L2RE ~ 1e-2."
> Implication: you need to drive the loss very low for useful accuracy. Don't stop at "loss looks flat."

**L-BFGS stalls but gradient is still useful** (Rathore et al., credence ~80%):
> "L-BFGS stops without reaching a critical point: gradient norm is ~1e-2 to 1e-3. The gradient still contains useful information."
> Cause: strong Wolfe line search fails, step size goes to zero.
> Fix: switch to NNCG (Armijo only) or restart with different LR.

**Condition number grows with nres** (Rathore et al., credence ~70%):
> Theorem 8.4: condition number = Omega(nres^alpha) where alpha > 1/2.
> With typical nres = 1e3 to 1e4, condition numbers > 1e4 are expected.
> Implication: more collocation points doesn't just mean more compute -- it makes the optimization harder.

---

## 4. Loss Design and Gradient Pathologies

### The core problem

The PINN loss has multiple terms (PDE residual, BCs, ICs, data) with different gradient magnitudes. The PDE residual term dominates because it involves differential operators that amplify gradients.

**L1 vs L2 norm choice** (Brunton 2023):
> L2 norm (MSE) on residuals: default; promotes smooth, low-frequency solutions. L1 norm (MAE) on residuals: more robust to outlier collocation errors and sharp gradients (shocks) since it doesn't square-penalize large pointwise residuals. This is distinct from L1 *regularization on equation coefficients*, which is what SINDy and sparse equation discovery use to promote parsimony (few active terms). Don't conflate the two: L1 residual = robust fitting; L1 coefficient regularization = sparse model selection. For standard PINNs with a known PDE, L2 is correct. L1 residual loss is worth trying if you have shocks or suspect outlier collocation points.
> Source: Brunton, S. "AI/ML+Physics Part 4 - Crafting a Loss Function." https://www.youtube.com/watch?v=3SNkQ8jhKXc

> Wang et al. 2021 (credence ~80%): "We observe that the gradient of the PDE residual loss is several orders of magnitude larger than the gradient of the boundary/initial condition losses." Demonstrated via histograms of per-parameter gradient magnitudes.
> Source: https://arxiv.org/abs/2001.04536, Figures 2-3

**Consequences:**
- BC/IC losses are undertrained (gradient signal drowned out)
- Solution satisfies the PDE approximately but violates BCs
- This looks like "the PINN found a wrong solution" but it's really "the optimizer is ignoring the BCs"

### Hessian stiffness

> Wang et al. 2021: Hessian eigenvalue ratios ~1e5 (max eigenvalue / min eigenvalue). This is the definition of an ill-conditioned problem.
> Source: https://arxiv.org/abs/2001.04536, Figures 4-5

### Solutions (in order of preference)

**1. Nondimensionalize first** (see Section 1). This is the cheapest fix and addresses the root cause.

**2. Learning rate annealing** (Wang et al. 2021, credence ~75%):
> Adaptively weight each loss term inversely proportional to its gradient magnitude. EMA of gradient statistics for stability.
> Source: https://arxiv.org/abs/2001.04536, Algorithm 1
> NeuralPDE.jl implements this as `GradientScaleAdaptiveLoss`.

**3. Conflict-free gradient methods** (ConFIG, credence ~70%):
> Instead of summing loss gradients (which can cancel), project them into a conflict-free direction.
> ConFIG: unit-normalize per-loss gradients, solve least-squares for combined direction, rescale by projection lengths.
> Source: https://tum-pbs.github.io/ConFIG/
> Key: must compute per-loss gradients separately (zero_grad + backward for each). Summing raw losses defeats the purpose.
> M-ConFIG: momentum variant, updates only one loss's gradient per step. Use with SGD, not Adam (momentum conflict).

**4. Don't use multiple losses if you can avoid it.** A single well-posed loss is always better than a weighted sum. Can you reformulate BCs as hard constraints (e.g., multiply network output by a function that satisfies BCs)? Can you use a penalty method that naturally balances?

**4b. Constrained optimization instead of penalized** (Brunton 2023, credence ~80%):
> Standard PINNs use penalized (soft) constraints: add physics as a loss term. The alternative is constrained optimization: minimize data error while exactly satisfying the physics constraints. "With a loss function you're not exactly satisfying your constraints. With constrained optimization you are."
> Source: Brunton, S. "AI/ML+Physics Part 5 - Employing an Optimization Algorithm." https://www.youtube.com/watch?v=T4iJ10TAIMg
> Physics-informed DMD (Baddoo et al. 2021) is the cleanest example: restrict the DMD matrix to a symmetry-preserving manifold (Hermitian, symplectic, etc.) via the Procrustes problem. KKT closed-form solutions exist because DMD is linear in its parameters -- the constraint is linear in both the output and the parameters simultaneously.
> Baddoo et al. 2021. "Physics-informed dynamic mode decomposition." Proc. R. Soc. A. https://arxiv.org/abs/2112.04307
> **Critical caveat for PINNs**: A BC like u(0)=0 is affine in the output u, but it is nonlinear in the NN weights theta. Closed-form KKT does NOT apply to neural network parameters. For NN-based PINNs, the two options for hard constraints are: (a) architectural -- multiply output by a distance function that satisfies the BC (Section 4 item 8), or (b) Augmented Lagrangian Methods (ALM), which are iterative and substantially more complex than Adam. Constrained optimization is most practical for linear models (DMD, SINDy, linear state-space) where the parameters enter linearly.

**5. Curriculum regularization** (Krishnapriyan et al. 2021 NeurIPS, credence ~80%):
> When the PINN fails on hard PDE regimes (high convection coefficient, strong reaction), don't start there. Start with easy parameters (small coefficient), train to convergence, then warm-start and increase to the target regime. 1-2 orders of magnitude improvement over naive training.
> "The curriculum training approach achieves significantly better errors, as well as lower variance in the error." (From Figure E.2 showing 10 seeds)
> Source: https://arxiv.org/abs/2109.01050, Sections 5.1 and Figure 4
> Evidence: evidence/krishnapriyan2021_failure_modes.md

**6. Sequence-to-sequence (time-marching)** (Krishnapriyan et al. 2021, credence ~75%):
> For time-dependent PDEs: train on a short time window, predict next state, step forward. Don't train on full space-time at once. "Posing the problem as seq2seq learning results in significantly lower error. The difference is particularly striking for reaction and reaction-diffusion cases, where seq2seq decreases error by almost two orders of magnitude."
> Source: https://arxiv.org/abs/2109.01050, Section 5.2
> NeuralPDE.jl calls this time-marching; see `WeightedIntervalTraining`.
> Note: these failures are not due to limited NN expressivity -- the architecture has enough capacity. The problem is optimization difficulty from the soft PDE constraint.

**7. Causal training** (Wang et al. 2022, credence ~85%):
> Standard PINNs trained by gradient descent are implicitly biased toward minimizing residuals at *later* times before even fitting the initial conditions -- violating physical causality. The NTK analysis shows the residual at time t is influenced more by residuals at later t' > t than earlier ones. This makes PINNs fail on chaotic/turbulent systems.
> Fix: weight each temporal residual point by wi = exp(-epsilon * sum_j<i R_j(theta)), where R_j is the accumulated residual before time i. This forces earlier times to converge first before the loss "turns on" at later times.
> "10-100x improvements in accuracy compared to competing approaches. First time PINNs succeeded on chaotic Lorenz, Kuramoto-Sivashinsky, and 2D Navier-Stokes in turbulent regime."
> Source: https://arxiv.org/abs/2203.07404, Abstract and Section 3
> Evidence: evidence/wang2022_causal_training.md
> Key difference from seq2seq/curriculum: causal weighting works within a single continuous training, without requiring separate time windows or changing the PDE coefficients. Can be combined with seq2seq for further gains.
> Sensitivity: epsilon controls the steepness of the causal weights. Too small = residuals at later times turn on too early. Too large = training stalls on early time steps. Anneal epsilon during training.

**8. Hard BCs via distance functions** (Sukumar & Srivastava 2022 CMAME, credence ~80%):
> Instead of penalizing BC violations (soft), multiply the PINN output by a distance function phi(x) that is zero on the boundary. Then u(x) = phi(x) * NN(x) satisfies BCs exactly by construction.
> "We eliminate modeling error associated with the satisfaction of boundary conditions. The sole contribution to the loss function is from the residual error at interior collocation points."
> "The proposed approach consistently outperforms a standard PINN-based collocation method."
> Source: https://arxiv.org/abs/2104.08426, Abstract and Section 1
> Evidence: evidence/sukumar2022_exact_bc_distance.md
> For our heat exchanger: u_{r,tube}(z,r) = r(R-r) * NN(z,r) enforces zero radial velocity at r=0 and r=R by construction. Same principle.

### Known failure modes

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

## 5. Sampling and Collocation

From NeuralPDE.jl:

| Strategy | When to use |
|---|---|
| GridTraining(dx) | Small domains, smooth solutions |
| StochasticTraining(N) | Default, re-sample each epoch |
| QuasiRandomTraining(N) | Better coverage (Latin hypercube, Sobol) |
| QuadratureTraining | Smooth, low-dimensional PDEs |
| WeightedIntervalTraining | Stiff ODEs, emphasize hard sub-intervals |

**Adaptive sampling** (residual-based): Sample more points where PDE residual is large. Helps with sharp features (phase change, shocks). But adds complexity.

**Smoothness bias**: Monte Carlo approximation of the integral loss naturally biases solutions toward smooth functions (low-frequency modes dominate). This is partly regularization (helpful) and partly harmful (misses sharp features like phase-change fronts, shocks). Mitigations: residual-adaptive sampling, curriculum regularization starting with smooth regime, Fourier feature embeddings to reduce spectral bias.

For 2D problems with radial integrals: use a regular grid in r (including r=0 and r=R) so torch.trapezoid / numpy.trapz can compute accurate integrals for conservation checks.

**How many points?** Start with ~1000 collocation points. More helps accuracy but worsens conditioning (Section 3). The sweet spot depends on the problem.

---

## 6. Property Mappings (EoS for PINNs)

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

## 7. Initial Conditions and Multi-Episode Training

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

## 8. Validation and General ML Debugging

### PINN-specific checks

1. **Energy conservation**: |Q_in - Q_out| / Q_total < 1%. For 2D: check per axial slice.
2. **Residual magnitude**: PDE residual should be near zero everywhere, not just on average. Plot residual vs (z, r).
3. **BC satisfaction**: Check that BCs are actually satisfied. The BC loss being small doesn't mean the BC is satisfied if the residual loss dominates the gradient.
4. **Physical reasonableness**: Does the temperature profile make physical sense? Is T_tube monotonically increasing? Is T_shell monotonically decreasing? Are there unphysical oscillations?
5. **Compare to classical solver**: Solve the same problem with scipy solve_bvp. If the PINN disagrees, the PINN is probably wrong.

### General ML debugging (from ml_debug folklore)

These apply to PINNs too:

**Work in order, don't skip to hyperparameters:**
1. Verify components in isolation (forward pass shapes, loss computation by hand, data pipeline)
2. Get signs of life on a toy problem (1D, known solution, constant Cp)
3. Overfit to training data first. If you can't overfit, you can't generalize.
4. Log everything: losses per component, gradient norms per module, parameter norms, activation stats
5. Numerical hygiene: `assert torch.isfinite(loss)`, `log(x.clamp(min=1e-8))`, `x / (std + 1e-5)`

**Symptom table** (adapted for PINNs):

| Symptom | Likely cause |
|---|---|
| Loss stuck from the start | LR too low, bad init, wrong loss function, nondimensionalization missing |
| Loss decreases then explodes | LR too high, numerical instability in T(h) near phase boundary |
| Loss NaN | log(0), 0/0 in property mapping, temperature outside EoS range |
| Physics loss low but BCs violated | Gradient imbalance (Section 4), need loss weighting |
| U converges to 0 or bound | Degenerate solution (Section 4), check sign convention |
| Model outputs constant | Dead neurons, vanishing gradients, U -> 0 |
| Good train, bad test RMSE | Overfitting to collocation points, not enough physics constraint |
| PINN worse than pure-data MLP | Wrong equations, bad scaling, or physics constraint fighting the data |

**Gradient clipping masks problems.** Always log the pre-clip norm. If it's constantly triggered, fix the root cause (nondimensionalize, reduce LR, check for numerical issues).

**Assume you have a bug.** Most of the time when a PINN doesn't work, it's a bug (sign error, wrong BC, missing factor of 2pi), not a hyperparameter issue.

### Loss surface analysis (from ml_debug)

When a loss term isn't working, don't guess -- visualize:
1. Grid over 1-2 key axes (e.g., U and a temperature), compute loss + gradient at each point.
2. Plot contour + quiver (negative gradient = optimization direction).
3. Look for: dead zones (zero gradient), saddle points, competing gradients, scale imbalance.

This takes 5 minutes and saves hours.

---

## 9. Multi-Loss Training Details (ConFIG)

If you must use multiple loss terms (and in PINNs you usually must):

### ConFIG (recommended over naive summation, credence ~70%)

```python
# Per-loss gradient capture (ConFIG requires this)
for loss_fn in [L_pde, L_bc, L_data]:
    optimizer.zero_grad()
    loss = loss_fn(model, collocation_pts)
    loss.backward()
    grads.append(get_gradient_vector(model))

# Combine via conflict-free direction
combined_grad = ConFIG_update(grads, use_least_square=True)
apply_gradient_vector(model, combined_grad)
optimizer.step()
```

**Key rules:**
- Don't pass summed loss into ConFIG. That defeats the purpose.
- Use `lstsq` not `pinv` (more stable).
- If some parameters don't participate in a loss, use `none_grad_mode='zero'`.
- M-ConFIG (momentum variant): use SGD, NOT Adam. Momentum conflict.

### Alternatives

| Method | Pros | Cons |
|---|---|---|
| Naive sum | Simple | Gradient conflict, dominant terms drown others |
| GradientScaleAdaptiveLoss | Built into NeuralPDE.jl | Heuristic, EMA lag |
| ReLoBRaLo | Effective on benchmarks | More complex |
| ConFIG | Theoretically grounded, consistent wins | Per-loss backward required (2-3x cost) |
| Hard constraints | Eliminates BC loss entirely | Not always possible |

> ConFIG authors claim superiority over PCGrad and Adam baseline on Burgers, Schrodinger, Kovasznay, Beltrami. UPGrad mentions ConFIG but ConFIG does not cite UPGrad.
> Source: https://tum-pbs.github.io/ConFIG/

---

## 10. Domain Decomposition (XPINNs)

For large or geometrically complex domains, split into subdomains each with a local PINN. Interface conditions enforce continuity between subdomains.

> Jagtap et al. 2020. "Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations." Commun. Comput. Phys. https://arxiv.org/abs/2005.11025
> Credence ~70%: Multiple citations, implemented in DeepXDE. Enables parallelization; each subdomain network is smaller and easier to optimize.
> Key: interface residuals must be added as additional loss terms. Continuity of u and its normal derivative across interfaces.
> Relevant for our heat exchanger: could split axially (cold end / hot end) if the solution has different character in different regions.

**Physics reparameterization**: Instead of learning u(x,t) from scratch, solve for the deviation from a known approximate solution: u(x,t) = u_approx(x,t) + delta(x,t), where the NN learns delta. If u_approx is good, delta is small and the landscape is smoother. The approximate solution can be the steady-state solution, a linearization, or a coarse numerical solution.

## 11. PIKANs (Kolmogorov-Arnold Networks for PINNs)

> Toscano et al. 2024: PIKANs "lead to smaller models and may also contribute to lowering computational cost while maintaining good accuracy."
> Source: https://arxiv.org/abs/2410.13228
> Credence ~40%: New, no independent replication. Other authors focus on improving PINNs within the MLP framework, not validating PIKANs as an alternative. Interesting but unproven.

---

## References

### Comprehensive guides (start here)
- Wang et al. 2023. "An Expert's Guide to Training Physics-Informed Neural Networks." arXiv:2308.08468. https://arxiv.org/abs/2308.08468
  - Key: most thorough practical guide. Covers architecture (modified MLP), sampling, loss weighting, Fourier features, causal training, code. By the same group as 2021 paper.
- Wang et al. 2021. "Understanding and mitigating gradient pathologies in physics-informed neural networks." SIAM J. Sci. Comput. https://arxiv.org/abs/2001.04536
  - Key: gradient imbalance diagnosis, learning rate annealing, modified MLP architecture

### Loss landscape and optimization
- Rathore et al. 2024. "Challenges in Training PINNs: A Loss Landscape Perspective." ICML. https://arxiv.org/abs/2402.01868
  - Key: Adam+L-BFGS, ill-conditioning from differential operators, near-zero loss required
- Krishnapriyan et al. 2021. "Characterizing possible failure modes in physics-informed neural networks." NeurIPS. https://arxiv.org/abs/2109.01050
  - Key: curriculum regularization, seq2seq, failure is optimization not expressivity

### Training strategies
- Wang et al. 2022. "Respecting causality for training physics-informed neural networks." J. Comput. Phys. https://arxiv.org/abs/2203.07404
  - Key: causal weighting of temporal residuals, 10-100x improvement on chaotic systems, first PINN success on turbulence
- Jagtap et al. 2020. "Extended Physics-Informed Neural Networks (XPINNs)." Commun. Comput. Phys. https://arxiv.org/abs/2005.11025
  - Key: domain decomposition, parallelizable, implemented in DeepXDE

### Hard constraints and boundary conditions
- Sukumar & Srivastava 2022. "Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks." CMAME. https://arxiv.org/abs/2104.08426
  - Key: distance-function trial functions, eliminates BC loss, consistently outperforms soft BCs
- Lagaris et al. 1998. "Artificial neural networks for solving ordinary and partial differential equations." IEEE Trans. Neural Netw. doi:10.1109/72.712178
  - Key: original paper on hard BCs via trial functions. Sukumar 2022 is the modern extension.

### Architecture and parameterization
- Wang et al. 2022. "Random Weight Factorization Improves the Training of Continuous Neural Representations." https://arxiv.org/abs/2210.01274
  - Key: factorize w = s * w_unit, better local minima, used in PirateNet
- Toscano et al. 2024. "From PINNs to PIKANs." https://arxiv.org/abs/2410.13228
- PirateNet / jaxpi: https://github.com/PredictiveIntelligenceLab/jaxpi (bundles RWF + causal + seq2seq + Fourier)
- Ling et al. 2016. "Machine learning strategies for systems with invariance properties." J. Comput. Phys. https://doi.org/10.1016/j.jcp.2016.05.003
  - Key: tensor-layer architecture enforcing Galilean invariance by construction for turbulence closure; symmetry-via-architecture beats symmetry-via-augmentation

### Constrained optimization and physics-informed DMD
- Baddoo et al. 2021. "Physics-informed dynamic mode decomposition." Proc. R. Soc. A. https://arxiv.org/abs/2112.04307
  - Key: restrict DMD to symmetry-preserving matrix manifolds (Hermitian, symplectic) via Procrustes problem; exactly satisfies conservation laws without penalty terms

### Lecture series
- Brunton, S. 2023. "AI/ML + Physics" YouTube lecture series. University of Washington.
  - Part 1 (problem definition): https://www.youtube.com/watch?v=ARMk955pGbg
  - Part 2 (training data): https://www.youtube.com/watch?v=g-S0m2zcKUg
  - Part 3 (architecture): https://www.youtube.com/watch?v=fiX8c-4K0-Q
  - Part 4 (loss function): https://www.youtube.com/watch?v=3SNkQ8jhKXc
  - Part 5 (optimization): https://www.youtube.com/watch?v=T4iJ10TAIMg

### Alternative formulations
- Weinan E & Bing Yu 2018. "The Deep Ritz Method: A Deep Learning-Based Numerical Algorithm for Solving Variational Problems." Commun. Math. Stat. https://arxiv.org/abs/1710.00211
  - Key: energy minimization formulation instead of strong-form residuals. Better-conditioned for elliptic PDEs, can be easier to optimize since the loss is an energy (always positive, no cancellation).

### Multi-loss training
- Liu et al. 2024. "ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks." https://tum-pbs.github.io/ConFIG/
- Bischof & Kraus. "ReLoBRaLo: Relative Loss Balancing with Random Lookback."
- TorchJD (UPGrad + aggregation methods): https://torchjd.org/stable/docs/aggregation/

### General ML debugging
- Schulman 2017. "Nuts and Bolts of Deep RL Experimentation." http://joschu.net/docs/nuts-and-bolts.pdf
- Andy Jones 2021. "Debugging RL, Without the Agonizing Pain." https://andyljones.com/posts/rl-debugging.html

### NeuralPDE.jl
- Architecture, sampling, adaptive loss patterns from tests and docs.
- https://github.com/SciML/NeuralPDE.jl
