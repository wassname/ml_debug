Title: 2203.07404v1.pdf

URL Source: https://arxiv.org/pdf/2203.07404

Published Time: Mon, 23 Jan 2023 13:55:29 GMT

Number of Pages: 35

Markdown Content:
# RESPECTING CAUSALITY IS ALL YOU NEED FOR TRAINING PHYSICS -INFORMED NEURAL NETWORKS 

Sifan Wang 

Graduate Group in Applied Mathematics and Computational Science University of Pennsylvania Philadelphia, PA 19104 

sifanw@sas.upenn.edu 

Shyam Sankaran 

Department of Mechanical Engineering and Applied Mechanics University of Pennsylvania Philadelphia, PA 19104 

shyamss@seas.upenn.edu 

Paris Perdikaris 

Department of Mechanical Engineering and Applied Mechanics University of Pennsylvania Philadelphia, PA 19104 

pgp@seas.upenn.edu 

March 16, 2022 

## ABSTRACT 

While the popularity of physics-informed neural networks (PINNs) is steadily rising, to this date PINNs have not been successful in simulating dynamical systems whose solution exhibits multi-scale, chaotic or turbulent behavior. In this work we attribute this shortcoming to the inability of existing PINNs formulations to respect the spatio-temporal causal structure that is inherent to the evolution of physical systems. We argue that this is a fundamental limitation and a key source of error that can ultimately steer PINN models to converge towards erroneous solutions. We address this pathology by proposing a simple re-formulation of PINNs loss functions that can explicitly account for physical causality during model training. We demonstrate that this simple modification alone is enough to introduce significant accuracy improvements, as well as a practical quantitative mechanism for assessing the convergence of a PINNs model. We provide state-of-the-art numerical results across a series of benchmarks for which existing PINNs formulations fail, including the chaotic Lorenz system, the Kuramoto–Sivashinsky equation in the chaotic regime, and the Navier-Stokes equations in the turbulent regime. To the best of our knowledge, this is the first time that PINNs have been successful in simulating such systems, introducing new opportunities for their applicability to problems of industrial complexity. 

Keywords Deep learning · Partial differential equations · Computational physics · Chaotic systems 

## 1 Introduction 

Physics-informed neural networks (PINNs) have emerged as a promising framework for synthesizing observational data and physical laws across diverse applications in science and engineering [ 1, 2, 3, 4, 5, 6, 7, 8]. However, it is well known that PINNs often face severe difficulties and even fail to tackle problems whose solution exhibits highly nonlinear, multi-scale, or chaotic behavior [ 9, 10 ]. Over the last few years, a series of extensions to the original formulation of Raissi et al. [ 11 ] have been proposed with the sole goal of enhancing the accuracy and robustness of PINNs in tackling increasingly more challenging problems. Such extensions include, but are not limited to, novel optimization algorithms for adaptive training [ 12 , 13 , 14 , 15 ], adaptive algorithms for selecting batches of training data [ 16 , 17 ], novel network architectures [ 12 , 9, 18 , 19 , 20 ], domain decomposition strategies [ 21 , 22 ], new types of activation 

> arXiv:2203.07404v1 [cs.LG] 14 Mar 2022

A PREPRINT - M ARCH 16, 2022 functions [ 23 ], and sequential learning strategies [ 16 , 24 , 25 ]. Although these techniques have been successful in introducing some improvements in terms of trainability and accuracy, there still exists a vast suite of problems that remain elusive to PINNs. Examples of such problems include systems whose behavior exhibits strong non-linearity, broadband energy spectra, and high sensitivity to initial conditions, such as the chaotic Kuramoto-Sivishinski equation and the Navier-Stokes equations in the turbulent regime. These are not pathological corner cases, but cases that are extremely relevant across a multitude of realistic scenarios in science and engineering. Therefore, there is a pressing need for understanding why PINNs fall short in such scenarios, and how they can be improved in order to overcome the challenges that currently limit their success to relatively simple problems. Physical systems are known to possess an inherent causal structure. Consider for example a linear wave with some initial velocity that is spreading out with a speed c across a homogeneous medium [ 26 ]. It is well-understood that, although a part of the wave may lag behind (if there is an initial velocity), no part can travel faster than speed c. This assertion encapsulates the so-called principle of causality that dictates how local changes in the initial/boundary data of a spatio-temporal dynamical system is reflected in its corresponding states at later times [ 26 ]. Specific to hyperbolic partial differential equations (PDEs), such as the wave equation, this principle underpins the formulation of the method of characteristics [ 27 ] that provides a rigorous set of analytical and numerical tools for efficiently tackling initial value problems. Although characterizing how information propagates in general nonlinear PDEs is a challenging task, basic principles of causality such as temporal precedence and covariation (i.e. statistical dependency between variables that are generated by coupled time evolution) are still expected to hold. This causal structure is also clearly reflected in classical numerical methods, where a PDE is typically discretized in time by sequential algorithms which ensure that the solution at time t is fully resolved before approximating the solution at time t + ∆ t. Strikingly, this notion of temporal dependence is absent in most continuous-time PINNs formulations (see e.g. [ 28 , 29 , 30 , 21 , 12 , 13 , 23 ]). In fact, as we will see in section 3, continuous-time PINNs trained by gradient descent are implicitly biased towards first approximating PDE solutions at later times, before even resolving the initial conditions, therefore profoundly violating temporal causality. Consequently, it is no surprise that such formulations are fragile and often fail to simulate forward problems, especially in cases where the target solutions exhibit strong dependence on initial data (e.g. chaotic systems). Recent studies [ 16 , 24 , 25 ] have proposed remedies to this issue by empirically introducing sequential training strategies, yet a concrete justification of why such strategies appear to be effective is still missing. This work is focused on investigating the importance of respecting physical causality during the training of continuous-time PINNs. Our specific contributions can be summarized as: • We reveal an implicit bias suggesting that continuous-time PINNs models can violate causality, and hence are susceptible to converge towards erroneous solutions. • We put forth a simple re-formulation of PINNs loss functions that allows us to explicitly respect the causal structure that characterizes the solution of general nonlinear PDEs. • Strikingly, we demonstrate that this simple modification alone is enough to introduce significant accuracy improvements, allowing us to tackle problems that have remained elusive to PINNs. • We provide a practical quantitative criterion for assessing the training convergence of a PINNs model. • We examine a collection of challenging benchmarks for which existing PINNs formulations fail, and demon-strate that the proposed causal training strategy leads to state-of-the-art results. To the best of our knowledge, this is the first time that PINNs have been successful in simulating systems such as the chaotic Lorenz system, the Kuramoto–Sivashinsky equation in the chaotic regime, and the Navier-Stokes equations in the turbulent regime, introducing new opportunities for their applicability to problems of industrial complexity. The paper is structured as follows. In section 2, we provide an overview of PINNs following the original formulation of Raissi et. al. [ 11 ]. Using a simple case study, we reveal an implicit bias of continuous-time PINNs that makes them prone to violating physical causality, and thereby steering them towards erroneous solutions. To address this drawback, in section 3 we put forth a simple re-formulation of the PINNs residual loss and propose a general casual training 

algorithm for explicitly respecting physical causality during model training. Section 4 discusses practical considerations specific to enhancing the accuracy and efficiency of PINNs. These developments are put to test in section 5, where we demonstrate state-of-the-art results across a comprehensive collection of challenging benchmarks for which existing PINN formulations are known to fail. Finally, section 6 provides a summary of our main findings, and touches upon remaining limitations and areas for future research. 2A PREPRINT - M ARCH 16, 2022 

## 2 Physics-informed neural networks (PINNs) 

Problem setup: We begin with a brief overview of physics-informed neural networks (PINNs) [ 11 ] in the context of inferring the solutions of PDEs. Generally, we consider PDEs taking the form 

ut + N [u] = 0 , t ∈ [0 , T ], x ∈ Ω, (2.1) subject to the initial and boundary conditions 

u(0 , x) = g(x), x ∈ Ω, (2.2) 

B[u] = 0 , t ∈ [0 , T ], x ∈ ∂Ω, (2.3) where N [·] is a linear or nonlinear differential operator, and B[·] is a boundary operator corresponding to Dirichlet, Neumann, Robin, or periodic boundary conditions. In addition, u describes the unknown latent solution that is governed by the PDE system of Equation 2.1. Following the original work of Raissi et al. [11 ], we proceed by representing the unknown solution u(t, x) by a deep neural network uθ (t, x), where θ denotes all tunable parameters of the network (e.g., weights and biases). Then, a physics-informed model can be trained by minimizing the following composite loss function 

L(θ) = λic Lic (θ) + λbc Lbc (θ) + λr Lr (θ), (2.4) where 

Lic (θ) = 1

Nic Nic ∑

> i=1

∣∣uθ (0 , xiic ) − g(xiic )∣∣2 , (2.5) 

Lbc (θ) = 1

Nbc Nbc ∑

> i=1

∣∣B[uθ ]( tibc , xibc )∣∣2 , (2.6) 

Lr (θ) = 1

NrNr∑

> i=1

∣∣∣∣

∂uθ

∂t (tir , xir ) + N [uθ ]( tir , xir )

∣∣∣∣

> 2

. (2.7) Here {xiic }Nic 

> i=1

, {tibc , xibc }Nbc  

> i=1

and {tir , xir }Nr 

> i=1

can be the vertices of a fixed mesh or points that are randomly sampled at each iteration of a gradient descent algorithm. Notice that all required gradients with respect to input variables or network parameters θ can be efficiently computed via automatic differentiation [31]. Moreover, the hyper-parameters 

{λic , λ bc , λ r } allow the flexibility of assigning a different learning rate to each individual loss term in order to balance their interplay during model training. These weights may be user-specified or tuned automatically during training [12, 13]. 

An illustrative example: To motivate the proposed methods described in section 3, let us study a representative case with which conventional PINN models are known to struggle. To this end, consider the one-dimensional Allen-Cahn equation 

ut − 0.0001 uxx + 5 u3 − 5u = 0 , t ∈ [0 , 1] , x ∈ [−1, 1] , (2.8) 

u(x, 0) = x2 cos( πx ), (2.9) 

u(t, −1) = u(t, 1) , (2.10) 

ux(t, −1) = ux(t, 1) . (2.11) This example is difficult to directly solve with the original continuous-time formulation of Raissi et al. [ 11 ], and has been recently studied by Wight et. al. [ 16 ] and McClenny et. al. [ 14 ] who developed adaptive re-sampling and weighting algorithms, respectively, to improve the PINNs prediction. Following the setup discussed in these studies [ 14 , 16 ], we represent the latent variable u by a fully-connected neural network uθ with tanh activation function, 4 hidden layers and 128 neurons per hidden layer. To further simplify the training objective 2.4, we also strictly impose the periodic BCs by embedding the input coordinates into Fourier expansion using Equation 4.8 with m = 10 (see section 4 for further details). Then the loss function 2.4 can be reduced to 

L(θ) = λic Lic (θ) + λr Lr (θ), (2.12) where Lic (θ) and Lr (θ) are defined exactly the same as in Equation 2.5 and Equation 2.7. For simplicity, we create a uniform mesh of size 100 × 256 in the computational domain [0 , 1] × [−1, 1] , yielding Nic = 256 initial points and 3A PREPRINT - M ARCH 16, 2022 

Nr = 25600 collocation points for enforcing the PDE residual. We also choose λic = 100 , λ r = 1 for better enforcing the initial condition. We proceed by training the resulting PINN model via full-batch gradient descent using the Adam optimizer [ 32 ] for 

2 × 10 5 iterations. As shown in Figure 1, even when the periodic boundary conditions are enforced exactly, our conventional PINN model is unable to learn the accurate solution for this example. One can also observe that the predicted solution seems to get stuck at some intermediate state and cannot be further refined to provide an accurate approximation to the ground truth. This is consistent with the left panel of Figure 2 where the loss functions rapidly decrease in the first few thousand training iterations, and then barely change for the rest of training, implying that the neural network gets trapped in an erroneous local minimum. Unfortunately, such problematic behavior is not a rare event, but rather a common outcome for PINNs, especially when solving transient problems [13, 24]. 

PINNs can violate physical causality: To explore the underlying reasons behind this failed case study, let us closely examine the definition of the residual loss Lr . Before doing so, we will slightly change our notation for convenience. Suppose that 0 = t1 < t 2 < · · · < t Nt = T discretizes the temporal domain, and {xj }Nx 

> j=1

discretizes the spatial domain Ω. For this example, {ti}Nt 

> i=1

and {xj }Nx 

> j=1

are uniformly spaced meshes in [0 , 1] and [−1, 1] , respectively. Now for a given spatial discretization {xj }Nx

> j=1

, we define the temporal residual loss as 

Lr (t, θ) = 1

NxNx∑

> j=1

| ∂uθ

∂t (t, xj ) + N [uθ ]( t, xj )|2. (2.13) Then, the residual loss 2.7 can be rewritten as 

Lr (θ) = 1

NtNt∑

> i=1

Lr (ti, θ) (2.14) 

= 1

NtNxNt∑

> i=1
> Nx

∑

> j=1

| ∂uθ

∂t (ti, xj ) + N [uθ ]( ti, xj )|2. (2.15) Next, we discretize ∂uθ 

> ∂t

using the forward Euler scheme [ 33 ]. For any 1 ≤ i ≤ Nt − 1, L(ti, θ) can be approximated by 

Lr (ti, θ) ≈ 1

NxNx∑

> j=1

∣∣∣∣

uθ (ti, xj ) − uθ (ti−1, xj )∆t + N [uθ ]( ti, xj )

∣∣∣∣

> 2

≈ |Ω|

∆t2

∫

> Ω

|uθ (ti, x) − uθ (ti−1, x) + ∆ tN [uθ ]( ti, x)|2dx. (2.16) From the above expression, we immediately obtain that the minimization of L(ti, θ) should be based on the correct prediction of both uθ (ti, x) and uθ (ti−1, x), while the original formulation of Equation 2.7 tends to minimize all 

L(ti, θ) simultaneously. As a result, by using Equation 2.7, the residual loss Lr (ti, θ) will be minimized even if the predictions at ti and previous times are inaccurate. This behavior inevitably violates temporal causality, making the PINN model susceptible to learn erroneous solutions. This conclusion is further confirmed by the middle panel of Figure 2 where we plot the temporal residual loss of Allen-Cahn equation at different iterations of training. As expected, the residual is quite large near the initial state and rapidly decays to nearly zero after t = 0 .5. We emphasize that the PDE temporal residual of small magnitude is meaningful only if the PINN model is well optimized and able to yield accurate predictions at the previous time steps. 

An undesirable implicit bias: To provide a deeper understanding of the fact that PINNs may violate temporal causality, we analyze their training dynamics through the lens of their empirical Neural Tangent Kernel (NTK) [ 34 , 13 ]. Specifically, for every Lr (t, θ) (Equation 2.13), we can define the empirical NTK Kθ (t) ∈ RNx×Nx whose ij -th entry is given by [13] 

Kθ (t)ij =

〈 ∂Rθ

∂θ (t, xi), ∂Rθ

∂θ (t, xj )

〉

, (2.17) where Rθ is the corresponding PDE residual defined by 

Rθ (t, x) = ∂uθ

∂t (t, x) + N [uθ ]( t, x), i, j = 1 , 2, . . . , N x. (2.18) 4A PREPRINT - M ARCH 16, 2022 −1 0 1

> x
> −1.0
> −0.50.0
> u(t, x )

t = 0   

> −101
> x
> −1.0
> −0.50.0
> u(t, x )

t = 0 .5  

> −101
> x
> −101
> u(t, x )

t = 1 .0

Figure 1: Allen-Cahn equation: Top: Reference solution versus the prediction of a trained conventional physics-informed neural network. The resulting relative L2 error is 49 .87% . Bottom: Comparison of the predicted and reference solutions corresponding to the three temporal snapshots at t = 0 .0, 0.5, 1.0.0.0 0.5 1.0 1.5 2.0               

> Iteration ×10 5
> 10 −4
> 10 −2
> 10 0
> Loss
> Lic
> Lr
> 0.00 0.25 0.50 0.75 1.00
> t
> 0.00.51.01.5
> L(t, θ )
> ×10 −1
> 0.00 0.25 0.50 0.75 1.00
> t
> 234
> C(t)
> ×10 3
> Iter = 1 ×10 3
> Iter = 1 ×10 4
> Iter = 1 ×10 5

Figure 2: Allen-Cahn equation: Left: Loss convergence of training a conventional physics-informed neural network for 2 × 10 5 iterations. Middle: Temporal residual loss L(t, θ) at different iteration of the training. Right: Temporal convergent rate at different iteration of the training. As demonstrated by Wang et. al. [ 13 ], the eigenvalues of Kθ (t) determine the convergence rate of each Lr (t, θ)

contributing to the total residual loss Lr (θ). Specifically, larger eigenvalues implies faster convergence rate. Following [13], we introduce the definition 

Definition 2.1. For any given t ∈ [0 , T ], the temporal convergence rate C(t) of Lr (t, θ) is defined by 

C(t) = 

∑Nt 

> k=1

λk(t)

Nt

= Trace (Kθ (t)) 

Nt

, (2.19) 

where {λk(t)}Nt 

> k=1

are the eigenvalues of Kθ (t).

Equipped with definition 2.19, we visualize C(t) at different iterations during the training of our PINNs model for solving Allen-Cahn equation. In the right panel of Figure 2, it can be seen that C(t) is greater if t is greater, indicating that the network is biased towards minimizing the temporal residual Lr (t, θ) for larger t. This reveals an undesirable implicit bias of continuous-time PINN models trained via gradient descent, suggesting that such models can profoundly violate the temporal causal structure that is inherent to time-dependent PDE systems. We argue that this inherent pathology of PINNs is the key underlying reason behind their inability to simulate transient problems that exhibit strong temporal correlations and sensitivity to initial data. In the next section we put forth a remarkably simple and effective strategy for explicitly respecting physical causality during the training phase PINNs. 5A PREPRINT - M ARCH 16, 2022 Method Relative L2 error Original formulation of Raissi et al. [11] 4.98 e − 01 

Adaptive time sampling [16] 2.33 e − 02 

Self-attention [14] 2.10 e − 02 

Time marching [25] 1.68 e − 02 

Causal training (MLP) 1.43e − 03 

Causal training (modified MLP) 1.39e − 04 

Table 1: Allen-Cahn equation: Relative L2 errors obtained by different approaches. 

## 3 Causal training for physics-informed neural networks 

A simple re-formulation: Based on our findings in the previous section, it is natural to ask how we can respect physical causality when solving PDEs with PINNs. We answer this question by introducing a simple re-formulation of the PINNs training objective that can explicitly account for the missing causal structure. To this end, we define a weighted residual loss as 

Lr (θ) = 1

NtNt∑

> i=1

wiLr (ti, θ). (3.1) We recognize that the weights wi should be large – and therefore allow the minimization of Lr (ti, θ) – only if all residuals {L r (tk, θ)}ik=1 before ti are minimized properly, and vice versa. This can be achieved by expressing the weights wi as 

wi = exp 

(

−

> i−1

∑

> k=1

Lr (tk, θ)

)

, for i = 2 , 3, . . . , N t, (3.2) where  will be referred to as a causality parameter that controls the steepness of the weights wi (see below for a more detailed discussion). As such, the weighted residual loss can be written as 

Lr (θ) = 1

NtNt∑

> i=1

exp 

(

−

> i−1

∑

> k=1

Lr (tk, θ)

)

Lr (ti, θ). (3.3) Notice that wi is inversely exponentially proportional to the magnitude of the cumulative residual loss from the previous time steps. As a consequence, Lr (ti, θ) will not be minimized unless all previous residuals {L r (tk, θ)}i−1 

> k=1

decrease to some small value such that wi is large enough. We now employ this simple modification and revisit the Allen-Cahn case study discussed before. We proceed by training the same network by minimizing the loss of Equation 2.4 using the weighted residual loss of Equation 3.3 with 

 = 100 , for 3 × 10 5 iterations of gradient descent under exactly the same hyper-parameter settings. The results of this experiment are summarized in Figure 3. One can see that the predicted solution achieves an excellent agreement with the ground truth, yielding an approximation error of 1.43 e − 03 measured in the relative L2 norm. The left panel of Figure 4 presents the convergence of the different loss function components, which is evidently much better than the one presented in Figure 2. Here we note that no other modifications between the two cases exist, besides the use of the proposed weighted residual loss of Equation 3.3. In fact, if in conjunction with the weighted residual loss we also employ a more powerful architecture for this example, such as the modified MLP [ 12 ] described in section 4, then we can achieve an even more accurate result with a resulting relative L2 error of 1.39 e − 04 . Additional detailed visual assessments for this example are provided in Appendix D. Finally, in Table 1 we provide the accuracy reported for this problem by existing approaches in the literature [ 14 , 16 , 25 ]. It is evident that the proposed methodology outperforms the best reported result of competing approaches by a factor of 

∼10-100x. This is a strong indication of the significance and necessity of respecting causality in training PINNs. 

A stopping criterion for assessing training convergence: To understand the effect of the residual weights {wi},we present the temporal residual loss and weights at different iterations of gradient descent in the middle and right panel of Figure 4 and Figure 16. We observe that the initial temporal weights are all zero except for t = 0 , implying that only 6A PREPRINT - M ARCH 16, 2022 −1 0 1

> x
> −1.0
> −0.50.0
> u(t, x )

t = 0   

> −101
> x
> −1.0
> −0.50.0
> u(t, x )

t = 0 .5  

> −101
> x
> −101
> u(t, x )

t = 1 .0

Figure 3: Allen-Cahn equation: Top: Reference solution versus the prediction of a trained physics-informed neural network using Algorithm 1. The resulting relative L2 error is 1.43 e − 03 . Bottom: Comparison of the predicted and reference solutions corresponding to the three temporal snapshots at t = 0 .0, 0.5, 1.0.0 1 2 3                 

> Iteration ×10 5
> 10 −7
> 10 −5
> 10 −3
> 10 −1
> Loss
> Lic
> Lr
> 0.00 0.25 0.50 0.75 1.00
> t
> 10 −5
> 10 −3
> 10 −1
> L(t, θ )
> 0.00 0.25 0.50 0.75 1.00
> t
> 0.00.20.40.60.81.0
> Temproal weights  w
> Iter = 0 Iter = 1 ×10 3
> Iter = 1 ×10 4
> Iter = 1 ×10 5
> Iter = 3 ×10 5

Figure 4: Allen-Cahn equation: Left: Loss convergence of training a physics-informed neural network using Algorithm 1. Middle: Temporal residual loss L(t, θ) at different iteration of the training. Right: Temporal weights at different iteration of the training. 

Lr (t0, θ) will be minimized at the beginning of training. Throughout the rest of the training, more temporal weights are activated, and eventually, all of them converge to 1 as the PDE residual loss is properly minimized. This last observation suggests that monitoring the magnitude of the residual weights {wi} can provide an effective stopping criterion for assessing the convergence of a PINNs model during training. Specifically, one can choose to terminate training of 

min i wi > δ , for some chosen threshold parameter δ ∈ (0 , 1) . As we will see in section 5, this stopping criterion not only helps to train a PINNs model faster, but it actually yields trained models with superior predictive accuracy. 

Sensitivity on the causality parameter : Here we must note that the results obtained using the proposed weighted residual loss do exhibit some sensitivity to the causality parameter  in Equation 3.2. Choosing a very small  can prevent the network from effectively minimizing the latter temporal residuals. On the other hand, choosing a large 

 value can result in a more difficult optimization problem, because the temporal residuals at earlier times have to decrease to a very small value in order to activate the latter temporal weights. This may be hard to achieve in some cases due to limited network capacity in minimizing the target residuals. In order to avoid tedious hyper-parameter tuning, we employ an annealing strategy for adjusting  using an increasing sequence of values {i}ki=1 , which gradually increases the strength with which the PDE residual constraint is enforced. As we will see in section 5, we empirically observe that this choice yields the best results in practice. 

Fitting the initial data: In the spirit of respecting causality, one may recognize that all temporal residuals should be minimized only if the network can first accurately fit the initial data. Therefore, we may treat the initial loss Lic as a special temporal residual at t = 0 and incorporate it into the weighted residual loss of Equation 3.1 in the same manner. 7A PREPRINT - M ARCH 16, 2022 

Causal training for PINNs: Based on the above remarks, Algorithm 1 presents a general causal training algorithm for PINNs. Specifically, it summarizes the proposed re-formulation of the residual and initial conditions loss, the annealing scheme for the  parameter, and the stopping criterion for terminating the training upon the convergence of the temporal weights wi. Accompanying Algorithm 1, here we present a few additional remarks worth discussing. 1. Although in this work we have limited our attention to PDEs with periodic boundary conditions that can be enforced in an exact manner (see section 4 for more details), the proposed causal training algorithm can be adapted to also incorporate boundary constraints using a similar treatment to the initial conditions loss. 2. Note that the temporal weights {wi}Nt 

> i=0

are a function of the trainable parameters θ. We use lax.stop_gradient 

in our JAX [35] implementation to prevent gradient back-propagation through the computation of wi.3. The computational cost of the proposed algorithm is negligible compared to conventional PINNs formulations since the weights wi are computed by directly evaluating the PINNs loss functions, whose values are already stored in the computational graph during training. 4. The proposed algorithm is not limited to fixed mesh points for evaluating the PINNs loss terms, and the collocation points can be randomly sampled at each iteration of gradient descent. The only requirement is that the sampled temporal points {ti}Nt 

> i=1

should form a non-decreasing sequence in temporal domain so that temporal causality can be respected. Here we should also mention that Algorithm 1 is general and can be employed within any existing physics-informed machine learning pipeline, including physics-informed neural networks [ 11 , 36 , 30 , 19 , 21 , 37 ], physics-informed deep operator networks [38, 39, 40], and physics-informed neural operators [41]. 

Algorithm 1: Causal training for physics-informed neural networks Consider a physics-informed neural network uθ (t, x) imposed the exact boundary conditions, and the corresponding weighted loss function 

L(θ) = 1

NtNt∑

> i=0

wiL(ti, θ), (3.4) where L(t0, θ) = λic Lic (θ) and for 1 ≤ i ≤ Nt, L(ti, θ) is defined in Equation 2.13. All wi are initialized by 1. Then use S steps of a gradient descent algorithm to update the parameters θ as: 

for  = 1, . . . ,  k do for n = 1 , . . . , S do 

(a) Compute and update the temporal weights by 

wi = exp 

(

−

> i−1

∑

> k=1

L(tk, θ)

)

, for i = 2 , 3, . . . , N t. (3.5) Here  > 0 is a user-defined hyper-parameter that determines the "slope" of temporal weights. (b) Update the parameters θ via gradient descent 

θn+1 = θn − η∇θ L(θn). (3.6) 

if min i wi > δ then 

break 

end end end 

The recommended hyper-parameters are λic = 10 3, δ = 0 .99 and {i}ki=1 = [10 −2, 10 −1, 10 0, 10 1, 10 2].

Connection to existing approaches: It is worth pointing out that the proposed residual weighting strategy bears some similarity to the adaptive time sampling of Wight et al. [16 ], since the effect of the weights wi can be viewed as equivalent to changing the sampling density of collocation points. However, the method of Wight et al. has two main disadvantages in practice: a) the sampling density has to be manually designed for different problems and training iterations, and b) an accurate approximation of the designed sampling density requires a large volume of collocation points, leading to a large computational cost. Besides, we remark that our method shares the same motivation with "time-marching" or "curriculum training" strategies [ 16 , 24 , 42 , 43 ], in the sense of respecting temporal causality by 8A PREPRINT - M ARCH 16, 2022 learning the solution sequentially within separate time-windows. In fact, our causal training strategy should not be viewed as a replacement of time-marching approaches, but instead as a crucial enhancement to those, given the fact that violations of causality may still occur within each time window of a time-marching algorithm. 

## 4 Practical considerations 

As we will see in section 5, high-order accuracy becomes a necessity for PINNs in order to tackle problems exhibiting sensitivity on initial data and strong spatio-temporal correlations (e.g. chaotic systems). Although PINNs are known for being incapable to achieve high-order accuracy in general, in this section we highlight a few extensions that can further enhance their performance in more challenging settings. Although these features are not deemed crucial for the successful application of Algorithm 1, we have empirically observed that, for the problems considered in this work, they can lead to further enhancements in terms of accuracy and computational efficiency. 

Modified multi-layer perceptrons: In [ 12 ] Wang et al. put forth a novel architecture that was demonstrated to outperform conventional MLPs across a variety of PINNs benchmarks. Here, we will refer to this architecture as "modified MLP". The forward pass of a L-layer modified MLP is defined as follows 

U = σ(XW 1 + b1), V = σ(XW 2 + b2), (4.1) 

H(1) = σ(XW (l) + b(l)), (4.2) 

Z(l) = σ(H(k)W (l) + b(l)), l = 1 , . . . , L − 1, (4.3) 

H(l+1) = (1 − Z(l)) U + Z(l) V , l = 1 , . . . , L − 1, (4.4) 

uθ (X) = H(L)W (L) + b(L), (4.5) where σ denotes a nonlinear activation function, denotes a point-wise multiplication, and X denotes an batch of input coordinates. All trainable parameters are given by 

θ = {W1, b1, W2, b1, (W (l), b(l))Ll=1 }. (4.6) At first glance, this architecture seems to appear a bit complicated. However, notice that it is almost the same as a standard MLP network, with the addition of two encoders and a minor modification in the forward pass. Specifically, the inputs X are embedded into a feature space via two encoders U , V , respectively, and merged in each hidden layer of a standard MLP using a point-wise multiplication. Based on our prior experience, the modified MLP architecture is shown to be more powerful than standard MLPs in terms of minimizing the PDE residuals and capturing sharp gradients [12, 9, 38, 39]. 

Exact periodic boundary conditions: Recent work by Dong et al. [44 ] showed how one can strictly impose periodic boundary conditions in PINNs as hard-constraints. We have empirically observed that this trick can simplify the training of PINNs and introduce some savings in terms of computational cost. To illustrate the main idea, let us consider enforcing periodic boundary conditions with period P in a one-dimensional setting. To this end, we would like to make sure that a neural network returns periodic predictions as 

u(l)(a) = u(l)(a + P ), l = 0 , 1, 2, . . . . (4.7) To enforce this constraint as part of the architecture itself, we construct a Fourier feature embedding of the form 

v(x) = (1 , cos( ωx ), sin( ωx ), cos(2 ωx ), sin(2 ωx ), · · · , cos( mωx ), sin( mωx )) , (4.8) with ω = 2πL , and some non-negative integer m. Then, for any network representation uθ , it can be proved that any 

uθ (v(x)) exactly satisfies the periodic constraint of Equation 4.7 (see [44] for a proof). The same idea can be extended to higher-dimensional domains. For instance, let (x, y ) denote the coordinates of a point in two dimensions, and suppose that u(x, y ) is a smooth periodic function to be approximated in a periodic cell 

[a, a + Px] × [b, b + Py ], satisfying the following constraints 

∂l

∂x l u (a, y ) = ∂l

∂x l u (a + Px, y ) , y ∈ [b, b + Py ] , (4.9) 

∂l

∂y l u (x, a ) = ∂l

∂y l u (x, b + Py ) , x ∈ [a, a + Px] , (4.10) 9A PREPRINT - M ARCH 16, 2022 for l = 0 , 1, 2, . . . , where Px and Py are the periods in the x and y directions, respectively. Similar to the one-dimensional setting, these constraints can be implicitly encoded in a neural network by constructing a two-dimensional Fourier features embedding as 

v(x, y ) = 



cos ( ωxx) cos ( ωy y) , . . . , cos ( nω xx) cos ( mω y y)cos ( ωxx) sin ( ωy y) , . . . , cos ( nω xx) sin ( mω y y)sin ( ωxx) cos ( ωy y) , . . . , sin ( nω xx) cos ( mω y y)sin ( ωxx) sin ( ωy y) , . . . , sin ( nω xx) sin ( mω y y)

 (4.11) with wx = 2πPx , w y = 2πPy and m, n being some non-negative integers. Following [ 44 ], any network representation 

uθ (v(x, y )) is guaranteed to be periodic in the x, y directions. For time-dependent problems, we simply concatenate the time coordinates t with the constructed Fourier features embedding, i.e., uθ ([ t, v(x)]) , or uθ ([ t, v(x, y )]) . Although in this work we will only consider periodic problems, other types of boundary conditions, including Dirichlet, Neumann, Robin, etc., can also be enforced in a "hard" manner, see [45, 46] for more details. 

Taylor-mode automatic differentiation for high-order derivatives: Conventional forward- or reverse-mode auto-matic differentiation is known to incur a cost that scales exponentially – both in terms of memory and computation – with the order of differentiation. This can quickly introduce a bottleneck in cases where derivatives of order higher than two are required (see e.g. the Kuramoto-Sivashinsky benchmark considered in section 5). To address this drawback, here we employ Taylor-mode automatic differentiation [ 31 ] in order to accelerate the computation of high-order derivatives. This is accomplished by leveraging a truncated Taylor polynomial approximation that allows for efficient computation of high-order derivatives of function compositions via the Faà di Bruno formula [31] 

∂n

∂x 1 · · · ∂x n

f (g(x)) = ∑

> σ∈π{1,...,n }

f (|σ|)(g(x)) ∏

> b∈σ

∂|b|

∏ 

> j∈b

∂x j

g(x), (4.12) where π{1, . . . , n } is the set of all partitions of the set {1, . . . , n }. It has been shown that Taylor-mode automatic differentiation enjoys much better scaling than conventional forward-mode or reverse-mode automatic differentiation, with its benefits becoming increasingly more dramatic as the order of differentiation is increased [ 47 ]. In terms of implementation, we leverage the jax.jet primitive accompanying the work of Bettencourt et al. [47, 35]. 

Parallel Training: Graphics processing units (GPUs) are the prevailing hardware choice for training PINNs, however these devices are often bound by their memory capacity. For more complex simulation scenarios (e.g. the Navier-Stokes benchmark in section 5) we have empirically observed that using larger batch sizes during training leads to enhanced convergence and predictive accuracy. However, a desirable batch size might exceed the available memory that a single GPU can offer, therefore motivating the use of data-parallelism across multiple GPU devices. In order to facilitate this, we utilize synchronous data-parallelism across multiple GPUs, with each GPU storing an identical copy of all trainable parameters. In this paradigm, a batch of training data is split into sub-batches, one for each device. Specifically, batches of spatial and temporal points used to evaluate the training loss are generated randomly and independently on each available GPU, and gradients of the training loss are then aggregated across all devices with a collective reduce-mean operation. As such, each device can then update its own local copy of all trainable model parameters at each gradient descent iteration the using global gradient signal that is broadcasted across all devices. In our implementation, this is efficiently performed leveraging the jax.pmap primitive in JAX [ 35 ], allowing us to seamlessly scale our code to an arbitrary number of GPUs. The parallel performance of our implementation will be assessed via strong and weak scaling studies, as discussed in section 5.3. 

## 5 Results 

Our goal in this section is to demonstrate the effectiveness of the proposed causal training algorithm by providing state-of-the-art numerical results for various types of differential equations exhibiting chaotic behavior, where existing PINNs formulations are destined for failure. Specifically, we will consider the forward simulation of the chaotic Lorenz system, the Kuramoto–Sivashinsky equation, and a two-dimensional simulation of decaying turbulence governed by the incompressible Navier-Stokes equations. Although these benchmarks can all be easily tackled using conventional numerical methods, they have remained elusive to PINNs since their initial conception [ 48 , 28 ], and all the variants that followed the reincarnation of this framework by Raissi et al. [29]. Throughout all benchmarks, we will employ the modified MLP architecture discussed in section 4 equipped with hyperbolic tangent activation functions (Tanh) and initialized using the Glorot normal scheme [ 49 ], unless otherwise 10 A PREPRINT - M ARCH 16, 2022 stated. We will enforce periodic boundary conditions as hard constraints by constructing appropriate Fourier features embedding of the input, as discussed in section 4. All networks are trained via stochastic gradient descent using the Adam optimizer with default settings [ 32 ] and an exponential learning rate decay with a decay-rate of 0.9 every 5, 000 

training iterations. As suggested by [ 16 , 24 , 25 ], we will also employ time-marching to reduce optimization difficulties. Specifically, we will split up the temporal domain of interest [0 , T ] into sub-domains [0 , ∆t], [∆ t, 2∆ t], . . . [T − ∆t, T ],and train networks to learn the solution in each sub-domain, where the initial condition is obtained from the prediction of the previously trained network. At the end of training, the resulting PINN model can produce predictions for the target solution at any continuous query location in the global spatio-temporal domain. All hyper-parameter settings, computational costs, implementation details and validation metrics are all discussed in the Appendix. The code and data accompanying this manuscript will be made publicly available at https: //github.com/PredictiveIntelligenceLab/CausalPINNs .

5.1 Lorentz system 

As our first example, we consider the chaotic Lorenz system. It is well known that this system exhibits strong sensitivity to its initial conditions, which can trigger divergent trajectories in finite time if the numerical predictions sought are not sufficiently accurate. The system is described by the following ordinary differential equations 

dx

dt = σ(y − x), (5.1) 

dy

dt = x(ρ − z) − y, (5.2) 

dz

dt = xy − βz. (5.3) These equations arise in studies of convection and instability in planetary atmospheric convection, where x, y, and z

denote variables proportional to convective intensity, horizontal, and vertical temperature differences [ 50 ]. Parameters 

ρ, σ and β denote the Prandtl number, Rayleigh number, and a geometric factor, respectively. The Lorenz system is well-known to be chaotic for certain parameter values and initial conditions. Here, we consider a classical setting with 

σ = 3 , ρ = 28 , and β = 8 /3. Our goal is to construct a PINNs model for learning the ODE solution up to time T = 20 ,starting from an initial condition [x(0) , y (0) , z (0)] = [1 , 1, 1] that does not lie on the system’s attractor. The employed PINN model architecture and training hyper-parameters are discussed in Appendix B. Figure 5 shows the predicted trajectory against the reference trajectory obtained via a classical numerical solver (see Appendix B for more details), where an excellent agreement can be observed with a relative L2 error 1.139 e −

02 , 1.656 e − 02 , 7.038 e − 03 for the x, y, z components, respectively. Moreover, all training losses are plotted in Appendix Figure 17. We can see that the stopping criterion min i wi > δ discussed in section 3 is satisfied for the training of each time window. It is worth pointing out that the proposed stopping criterion will not only benefit the predictive accuracy, but also save lots of computational costs. To verify this, we train the network by removing the stopping criterion and training for a fixed number of iterations for each time window under exactly the same hyper-parameter setting. Interestingly, as shown in Appendix 19, the training losses can achieve slightly lower values than the ones using the stopping criterion. However, the model predictions are less accurate, as some discrepancies can be clearly observed in Appendix Figure 18. Although the reason behind this behavior still remains unclear, it appears that training the model for more iterations after the proposed stopping criterion has been met seems to give rise to over-fitting. 

5.2 Kuramoto–Sivashinsky equation 

The next example aims to illustrate the effectiveness of our method in tackling spatio-temporal chaotic systems. To this end, we consider one-dimensional Kuramoto–Sivashinsky equation, which has been independently derived in the context of reaction-diffusion systems [ 51 ] and flame front propagation [ 52 ]. The Kuramoto–Sivashinsky equation exhibits a wealth of spatially and temporally nontrivial dynamical behavior including chaos, and has served as a model example in efforts to understand and predict the complex dynamical behavior associated with a variety of physical systems. The equation takes the form 

ut + αuu x + βu xx + γu xxxx = 0 , (5.4) subject to periodic boundary conditions and an initial condition 

u(0 , x ) = u0(x). (5.5) 11 A PREPRINT - M ARCH 16, 2022 0 5 10 15 20                            

> t
> −10 010 20
> Predicted x(t)
> 0510 15 20
> t
> −20
> −10 010 20
> Predicted y(t)
> 0510 15 20
> t
> 010 20 30 40 50 Predicted z(t)
> 0510 15 20
> t
> 10 −5
> 10 −3
> 10 −1
> Absolute error x(t)
> 0510 15 20
> t
> 10 −5
> 10 −3
> 10 −1
> Absolute error y(t)
> 0510 15 20
> t
> 10 −5
> 10 −3
> 10 −1
> Absolute error z(t)

Figure 5: Lorentz system: Comparison between the predicted and reference solutions. 

Case I (regular): We start with a relatively simple scenario by setting α = 5 , β = 0 .5, γ = 0 .005 , and a spatial domain [−1, 1] . The initial condition is given by u0(x) = − sin( πx ). Our goal is to lean the associated solution up to time T = 1 . A detailed visual assessment of the predicted solution is presented in Figure 6. In particular, we present a comparison between the reference and the predicted solutions at different time instants t = 0 , 0.5, 1.0. It can be observed that the PINNs prediction achieves an excellent agreement with the reference solutions, yielding an error of 

3.49 e − 04 measured in the relative L2 norm. This is further illustrated by the temporal relative L2 error shown in the left panel of Figure 8. Particularly, one may note that the error increases drastically by one order of magnitude for 

t ∈ [0 .4, 0.6] where the solution happens to experience a fast transition. This behavior is consistent with the larger loss values and the larger number of training iterations required before the stopping criterion is met, as observed in Appendix Figure 20. To highlight the computational efficiency of Taylor-mode automatic differentiation (Taylor-mode AD) discussed in section 4, here we provide a comparison in terms of computational cost against conventional reverse-mode automatic differentiation (AD) [ 31 ]. Specifically, we consider PINN models with a different number of layers and batch sizes. As shown in Figure 7, Taylor-mode AD provides a significant advantage in terms of computational efficiency, allowing us to accommodate larger architectures and batch sizes. As a consequence, for the same architecture and batch size, we have consistently observed a speed-up of 3-5x in the total training time required for Taylor-mode AD versus conventional AD. 

Case II (chaotic): We proceed by solving a more challenging case exhibiting chaotic behavior, which remains stubbornly unsolved using existing PINNs formulations [ 53 ]. Specifically, we set α = 100 /16 , β = 100 /16 2, γ =100 /16 4, for a fixed spatial domain in [0 , 2π]. Starting from an initial condition in the chaotic regime, we use PINNs to solve Kuramoto–Sivashinsky equation up to time T = 0 .5. The results are summarized in Figure 9, from which one can see that the predicted solution is in good agreement with the reference solution obtained via classical spectral methods (see Appendix F for more details). The resulting relative L2 error over the entire spatio-temporal domain is 

2.46 e − 02 , which is visualized in the right panel of Figure 8. These results highly suggest that the proposed causal training algorithm enables PINN models to capture the intricate chaotic behavior of this system. From a critical standpoint, here we should also mention that difficulties can still arise in simulating the long-time behavior of chaotic systems. Figure 10 summarizes our results starting with a simple initial state u0(x) = cos( x)(1 + sin( x)) ,and simulating the dynamics up to time T = 0 .9. One can observe that the predicted solution accurately captures the transition to chaos at around t = 0 .4, while eventually loses accuracy after t = 0 .8 due to the chaotic nature of the problem and the inevitable numerical error accumulation of PINNs, leading to a relative L2 error above 10% for the final state. This highlights the crucial need for further enhancing the accuracy of PINN approximations in order to retain effectiveness in such complex regimes. Long-time integration in general, has been one PINNs’ major drawbacks, and in future work we plan to address this via operator learning techniques as described in [39]. 12 A PREPRINT - M ARCH 16, 2022 −1 0 1       

> x
> −101
> u(t, x )
> t= 0
> −101
> x
> −202
> u(t, x )
> t= 0 .5
> −101
> x
> −202
> u(t, x )
> t= 1 .0

Figure 6: Kuramoto–Sivashinsky equation (regular): Top: Reference solution versus the prediction of a trained physics-informed neural network using Algorithm 1. The resulting relative L2 error is 3.49 e − 04 . Bottom: Comparison of the predicted and reference solutions corresponding to the three temporal snapshots at t = 0 , 0.5, 1.0.4 6 8 10 

# Layers 

0.25 0.50 0.75 1.00 

> Time ( ms )

×10 1

10 2 10 3 10 4

Batch size 

012

> Time ( ms )

×10 1 

> AD Taylor-mode AD

Figure 7: Kuramoto–Sivashinsky equation (regular): Left: Timing of evaluating the loss function of a PINN model with different number of layers. The rest hyper-parameters are the same as in Table 3. Right: Timing of evaluating the forward pass of a PINN model with different batch sizes. The rest hyper-parameters are the same as in Table 3. 0.0 0.5 1.0

t  

> 10 −5
> 10 −4
> 10 −3
> Rel.  L2 error 0.00.20.4

t

> 10 −3
> 10 −1
> Rel.  L2 error

Figure 8: Kuramoto–Sivashinsky equation: Left: Relative L2 errors of Case I (regular). Right: Relative L2 errors of Case II (chaotic). 13 A PREPRINT - M ARCH 16, 2022 0.0 2.5 5.0       

> x
> −202
> u(t, x )
> t= 0
> 0.02.55.0
> x
> −202
> u(t, x )
> t= 0 .25
> 0.02.55.0
> x
> −202
> u(t, x )
> t= 0 .5

Figure 9: Kuramoto–Sivashinsky equation (chaotic): Top: Reference solution versus the prediction of a trained physics-informed neural network using Algorithm 1. The resulting relative L2 error is 2.26 e − 02 . Bottom: Comparison of the predicted and reference solutions corresponding to the three temporal snapshots at t = 0 , 0.25 , 0.5.

Figure 10: Kuramoto–Sivashinsky equation (chaotic): Reference solution versus the prediction of a trained physics-informed neural network using Algorithm 1. The initial condition is u0(x) = cos( x)(1 + sin( x)) An animation of the solution evolution is provided at https://github.com/PredictiveIntelligenceLab/CausalPINNs# kuramotosivashinsky-equation .

5.3 Navier-Stokes equation 

To further emphasize the effectiveness of the proposed causal training algorithm for solving chaotic dynamical systems, in the last example, we consider a classical two-dimensional decaying turbulence example in a square domain with periodic boundary conditions. This problem can be modeled via the incompressible Navier-Stokes equations expressed in the velocity-vorticity formulation 

wt + u · ∇ w = 1

Re ∆w, in [0 , T ] × Ω, (5.6) 

∇ · u = 0 , in [0 , T ] × Ω, (5.7) 

w(0 , x, y ) = w0(x, y ), in Ω, (5.8) where u = ( u, v ) denotes the flow velocity field, w = ∇ × u denotes the vorticity, and Re is the Reynolds number. In addition, we set Ω = [0 , 2π]2 and Re = 100 . Our goal is to use PINNs to simulate the flow up to T = 1 .Figure 11 presents the predicted velocity and vorticity field at T = 1 . More detailed visual assessments are provided in Appendix G. We can see that all latent variables of interest are in good agreement with their corresponding reference solutions, yielding an error of 3.90 e−02 , 2.61 e−02 , 3.53 e−02 for u, v, w , respectively, over the entire spatio-temporal domain. This observation is further illustrated by the resulting errors reported in Figure 12 and the computed energy spectrum in Figure 13. These results highlight the remarkable effectiveness of the proposed causal training algorithm, successfully enabling the PINNs model to capture such complicated turbulent flow without any training data. 14 A PREPRINT - M ARCH 16, 2022 

Figure 11: Navier-Stokes equation: Representative snapshot of the predicted velocity and vorticity versus the corre-sponding reference solution at t = 1 . An animation of the solution evolution is provided at https://github.com/ PredictiveIntelligenceLab/CausalPINNs#navier-stokes-equation .0.0 0.5 1.0    

> t
> 10 −3
> 10 −2
> 10 −1
> Rel.  L2 error
> u(t, x, y )
> 0.00.51.0
> t
> 10 −3
> 10 −2
> Rel.  L2 error
> v(t, x, y )
> 0.00.51.0
> t
> 10 −3
> 10 −2
> 10 −1
> Rel.  L2 error
> w(t, x, y )

Figure 12: Navier-Stokes equation: Relative L2 prediction errors for u, v, w , respectively. For this benchmark, we also report the performance of our parallel JAX implementation on a compute node equipped with 8 NVIDIA Ampere A6000 GPUs. We use an effective batch-size of 42 , 000 spatio-temporal points sampled in each training iteration on each GPU with a network consisting of 6 layers with 300 neurons per layer. Figure 14 presents the scaling results obtained. To conduct a strong scaling study, we keep the problem size fixed and split the batch across several GPUs. As expected, we notice a speed-up, but the benefits deteriorate as the number of GPUs is increased beyond 4. We attribute this behavior to the fact that, for a fixed problem size, the compute load assigned to each GPU decreases as the number of devices is increased, leading to an under-utilization of each device. We have also performed a weak scaling study in which the number of points sampled per GPU is fixed. Under this setting, we report excellent parallel efficiency that remains above 99% as the number of GPUs is increased. While we have only considered data-parallelism in this study, we may be able to obtain further speed-ups by considering a combination of data- and function-parallelism techniques [ 54 ] in future studies. Figure 14 also reports the effect of batch-size of training on the resulting L2 accuracy for the first time-window ( t ∈ [0 , 0.1] ). In general, we notice that an increase in batch-size results in higher accuracy of the network. This motivates the use of larger batch sizes through data-parallelism as a mechanism for enhancing the accuracy of PINNs in more challenging problems. 15 A PREPRINT - M ARCH 16, 2022 10 0 10 1 10 2        

> k
> 10 −10
> 10 −6
> 10 −2
> E(k)/ ∑ E(k) k−4
> t= 0 .0
> 10 010 110 2
> k
> 10 −10
> 10 −6
> 10 −2
> E(k)/ ∑ E(k) k−4
> t= 0 .5
> 10 010 110 2
> k
> 10 −10
> 10 −6
> 10 −2
> E(k)/ ∑ E(k) k−4
> t= 1 .0
> Reference Predicted

Figure 13: Navier-Stokes equation: Reference versus predicted normalized kinetic energy spectra at different time snapshots t = 0 .0, 0.5, 1.0.1 2 3 4 5 6 7 8           

> # Devices
> 2468
> Speedup
> Strong Scaling
> 12345678
> # Devices
> 99 .499 .699 .8100 .0
> Parallel Efficiency (%)
> Weak Scaling
> 4816
> Nt
> 256 128 64 32 16
> Nx
> Rel. L2error
> 0.003 0.004 0.005 0.006 0.007 0.008

Figure 14: Parallel Performance: Left: Strong Scaling: Keeping the total-batch size for the problem fixed, we evaluate the speedup obtained when the batch is split across multiple devices. Centre: Weak Scaling: Keeping the batch-size on each GPU fixed, we report the efficiency of scaling by dividing the time taken on a single device over the time taken on 

n-devices. Right: Effect of batch-size: L2 error for models trained till t = 0 .1 using Nt and Nx points per iteration in the temporal and spatial domain respectively. 

## 6 Discussion 

Physical systems possess an inherent causal structure that explains the fundamental relationship between causes and effects governing their dynamic evolution. In this work, we show that physics-informed neural networks are prone to violating that structure when trained to infer the solution of time-dependent PDEs. Specifically, by studying the limiting neural tangent kernel of PINNs we reveal an implicit bias indicating a preference of PINNs to first minimize PDE residuals at later times, before even fitting the initial data. We argue that this fundamental drawback is one of the key reasons why PINNs can fail in practice. To resolve this shortcoming, we propose a novel causal training algorithm that can restore physical causality during the training of a PINNs model by appropriately re-weighting the PDE residual loss at each iteration of gradient descent. Interestingly, this also leads to a simple stopping criterion for effectively assessing the convergence of the total training loss. We demonstrate that this simple modification alone is sufficient to achieve 10-100x improvements in accuracy compared to competing approaches, opening the path to tackling challenging problems that were not accessible to PINNs before, such as the chaotic Lorenz and Kuramoto-Sivashinsky equations, and the incompressible Navier-Stokes equations in the turbulent regime. In this work we have solely focused on forward simulation problems, as we believe that these are the cases that most strongly expose the challenges and limitations in building PINNs models. While it is true that PINNs are currently better suited and have enjoyed far more success in tackling hybrid/inverse problems in which observational data is available, we believe that respecting causality is a crucial factor to consider when training a PINNs model, regardless of the forward/inverse nature of a given problem. To this end, in the inverse problem setting one should consider observational data as point sources of information, and ensure that PDE residuals are first adequately minimized at those locations before propagating information outwards. A more detailed exploration of this direction will be sought in future work. We must also note that different problems are likely to pose a different causal structure. For example, in optimal control one needs to predict the state of a system by evolving its dynamics forward in time from a given initial condition, but also compute sensitivities with respect to a control input by evolving the adjoint system backwards in time from a given terminal condition that depends on the final system state. In this case, what we here refer to as “temporal causality" takes a different form for the state (forward) and the co-state (adjoint) simulations. However, our main message remains the 16 A PREPRINT - M ARCH 16, 2022 same: respecting causality matters, and training algorithms for PINNs should be designed to respect how information propagates according to the underlying principles that govern the evolution of a given system. Given the rising prominence of PINNs across academic and industrial use cases, we consider this as a hallmark contribution that sets a new standard for what such models are capable of. We anticipate that the findings of this work will create new opportunities for the application of PINNs to more complex scenarios across diverse domains, including fluid mechanics, electromagnetics, quantum mechanics, and elasticity. However, despite the encouraging results reported here, there is a still gap between the current progress in PINNs research and real-world applications. We have to admit that viewing PINNs as a forward PDE solver is significantly more time-consuming than the traditional numerical solvers. Therefore, future research should focus on accelerating the training of PINNs. Distributed and parallel implementations can be of great help [ 55 , 21 ] in this direction. Another aspect with great room for improvement is related to architecture design. Even though effective modifications such as the modified MLP discussed in section 4 and in [ 12 ] can introduce noticeable gains in accuracy, a niche architecture similar to what convolutional networks have been for vision or Transformers for language processing, is yet to be discovered for solving PDEs. To this end, we must recognize that training a PINN model is fundamentally different from solving conventional supervised learning tasks, requiring us to design more effective architectures for minimizing PDE residuals in a self-supervised manner. We believe that addressing these open questions will become an important piece of the puzzle in advancing the use of physics-informed machine learning as a reliable analysis tool in computational science and engineering. 

## Acknowledgements 

We would like to acknowledge support from the US Department of Energy under the Advanced Scientific Computing Research program (grant DE-SC0019116), the US Air Force (grant AFOSR FA9550-20-1-0060), and US Department of Energy/Advanced Research Projects Agency (grant DE-AR0001201). We also thank the developers of the software that enabled our research, including JAX [35], JAX-CFD[56], Matplotlib [57], and NumPy [58]. 

## References 

[1] Maziar Raissi, Alireza Yazdani, and George Em Karniadakis. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science , 367(6481):1026–1030, 2020. [2] Abhilash Mathews, Manaure Francisquez, Jerry W Hughes, David R Hatch, Ben Zhu, and Barrett N Rogers. Uncov-ering turbulent plasma dynamics via deep learning from partial observations. Physical Review E , 104(2):025205, 2021. [3] Georgios Kissas, Yibo Yang, Eileen Hwuang, Walter R Witschey, John A Detre, and Paris Perdikaris. Machine learning in cardiovascular flows modeling: Predicting arterial blood pressure from non-invasive 4D flow MRI data using physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering , 358:112623, 2020. [4] Alireza Yazdani, Lu Lu, Maziar Raissi, and George Em Karniadakis. Systems biology informed deep learning for inferring parameters and hidden dynamics. PLoS computational biology , 16(11):e1007575, 2020. [5] Sifan Wang and Paris Perdikaris. Deep learning of free boundary and Stefan problems. Journal of Computational Physics , 428:109914, 2021. [6] Khemraj Shukla, Patricio Clark Di Leoni, James Blackshire, Daniel Sparkman, and George Em Karniadakis. Physics-informed neural network for ultrasound nondestructive quantification of surface breaking cracks. Journal of Nondestructive Evaluation , 39(3):1–20, 2020. [7] Yuyao Chen, Lu Lu, George Em Karniadakis, and Luca Dal Negro. Physics-informed neural networks for inverse problems in nano-optics and metamaterials. Optics express , 28(8):11618–11633, 2020. [8] Francisco Sahli Costabal, Yibo Yang, Paris Perdikaris, Daniel E Hurtado, and Ellen Kuhl. Physics-informed neural networks for cardiac activation mapping. Frontiers in Physics , 8:42, 2020. [9] Sifan Wang, Hanwen Wang, and Paris Perdikaris. On the eigenvector bias of fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering , 384:113938, 2021. [10] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics , pages 1–19, 2021. [11] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics , 378:686–707, 2019. 17 A PREPRINT - M ARCH 16, 2022 [12] Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing , 43(5):A3055–A3081, 2021. [13] Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics , 449:110768, 2022. [14] Levi McClenny and Ulisses Braga-Neto. Self-adaptive physics-informed neural networks using a soft attention mechanism. arXiv preprint arXiv:2009.04544 , 2020. [15] Suryanarayana Maddu, Dominik Sturm, Christian L Müller, and Ivo F Sbalzarini. Inverse dirichlet weighting enables reliable training of physics informed neural networks. Machine Learning: Science and Technology , 2021. [16] Colby L Wight and Jia Zhao. Solving Allen-Cahn and Cahn-Hilliard equations using the adaptive physics informed neural networks. arXiv preprint arXiv:2007.04542 , 2020. [17] Mohammad Amin Nabian, Rini Jasmine Gladstone, and Hadi Meidani. Efficient training of physics-informed neural networks via importance sampling. Computer-Aided Civil and Infrastructure Engineering , 2021. [18] Jie Bu and Anuj Karpatne. Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving PDEs. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM) , pages 675–683. SIAM, 2021. [19] Ameya D Jagtap, Yeonjong Shin, Kenji Kawaguchi, and George Em Karniadakis. Deep kronecker neural networks: A general framework for neural networks with adaptive activation functions. Neurocomputing , 468:165–180, 2022. [20] Senwei Liang, Liyao Lyu, Chunmei Wang, and Haizhao Yang. Reproducing activation function for deep learning. 

arXiv preprint arXiv:2101.04844 , 2021. [21] Ameya D Jagtap and George Em Karniadakis. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. Communications in Computational Physics , 28(5):2002–2041, 2020. [22] Ben Moseley, Andrew Markham, and Tarje Nissen-Meyer. Finite basis physics-informed neural networks (fbpinns): a scalable domain decomposition approach for solving differential equations. arXiv preprint arXiv:2107.07871 ,2021. [23] Ameya D Jagtap, Kenji Kawaguchi, and George Em Karniadakis. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics , 404:109136, 2020. [24] Aditi S Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M Kirby, and Michael W Mahoney. Characterizing possible failure modes in physics-informed neural networks. arXiv preprint arXiv:2109.01050 , 2021. [25] Revanth Mattey and Susanta Ghosh. A novel sequential method to train physics informed neural networks for allen cahn and cahn hilliard equations. Computer Methods in Applied Mechanics and Engineering , 390:114474, 2022. [26] Walter A Strauss. Partial differential equations: An introduction . John Wiley & Sons, 2007. [27] L.C. Evans and American Mathematical Society. Partial Differential Equations . Graduate studies in mathematics. American Mathematical Society, 1998. [28] Isaac E Lagaris, Aristidis Likas, and Dimitrios I Fotiadis. Artificial neural networks for solving ordinary and partial differential equations. IEEE transactions on neural networks , 9(5):987–1000, 1998. [29] Maziar Raissi, Hessam Babaee, and Peyman Givi. Deep learning of turbulent scalar mixing. Physical Review Fluids , 4(12):124501, 2019. [30] Ehsan Kharazmi, Zhongqiang Zhang, and George Em Karniadakis. Variational physics-informed neural networks for solving partial differential equations. arXiv preprint arXiv:1912.00873 , 2019. [31] Andreas Griewank and Andrea Walther. Evaluating derivatives: principles and techniques of algorithmic differentiation . SIAM, 2008. [32] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 ,2014. [33] Arieh Iserles. A first course in the numerical analysis of differential equations . Number 44. Cambridge university press, 2009. [34] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in neural information processing systems , pages 8571–8580, 2018. 18 A PREPRINT - M ARCH 16, 2022 [35] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transforma-tions of Python+NumPy programs, 2018. [36] Lu Lu, Xuhui Meng, Zhiping Mao, and George E Karniadakis. DeepXDE: A deep learning library for solving differential equations. arXiv preprint arXiv:1907.04502 , 2019. [37] Oliver Hennigh, Susheela Narasimhan, Mohammad Amin Nabian, Akshay Subramaniam, Kaustubh Tangsali, Zhiwei Fang, Max Rietmann, Wonmin Byeon, and Sanjay Choudhry. Nvidia simnet ™ : An ai-accelerated multi-physics simulation framework. In International Conference on Computational Science , pages 447–461. Springer, 2021. [38] Sifan Wang, Hanwen Wang, and Paris Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed DeepOnets. arXiv preprint arXiv:2103.10974 , 2021. [39] Sifan Wang and Paris Perdikaris. Long-time integration of parametric evolution equations with physics-informed deeponets. arXiv preprint arXiv:2106.05384 , 2021. [40] Sifan Wang, Hanwen Wang, and Paris Perdikaris. Improved architectures and training algorithms for deep operator networks. arXiv preprint arXiv:2110.01654 , 2021. [41] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. arXiv preprint arXiv:2111.03794 , 2021. [42] Yifan Du and Tamer A Zaki. Evolutional deep neural network. arXiv preprint arXiv:2103.09959 , 2021. [43] Shashank Reddy Vadyala, Sai Nethra Betgeri, and Naga Parameshwari Betgeri. Physics-informed neural network method for solving one-dimensional advection equation using pytorch. Array , 13:100110, 2022. [44] Suchuan Dong and Naxian Ni. A method for representing periodic functions and enforcing exactly periodic boundary conditions with deep neural networks. Journal of Computational Physics , 435:110242, 2021. [45] N Sukumar and Ankit Srivastava. Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. arXiv preprint arXiv:2104.08426 , 2021. [46] Lu Lu, Raphael Pestourie, Wenjie Yao, Zhicheng Wang, Francesc Verdugo, and Steven G Johnson. Physics-informed neural networks with hard constraints for inverse design. arXiv preprint arXiv:2102.04626 , 2021. [47] Jesse Bettencourt, Matthew J Johnson, and David Duvenaud. Taylor-mode automatic differentiation for higher-order derivatives in jax. 2019. [48] Dimitris C Psichogios and Lyle H Ungar. A hybrid neural network-first principles approach to process modeling. 

AIChE Journal , 38(10):1499–1511, 1992. [49] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 249–256, 2010. [50] Edward N Lorenz. Deterministic nonperiodic flow. Journal of atmospheric sciences , 20(2):130–141, 1963. [51] Yoshiki Kuramoto and Toshio Tsuzuki. Persistent propagation of concentration waves in dissipative media far from thermal equilibrium. Progress of theoretical physics , 55(2):356–369, 1976. [52] Gregory I Sivashinsky. Nonlinear analysis of hydrodynamic instability in laminar flames—i. derivation of basic equations. Acta astronautica , 4(11):1177–1206, 1977. [53] Maziar Raissi. Deep hidden physics models: Deep learning of nonlinear partial differential equations. The Journal of Machine Learning Research , 19(1):932–955, 2018. [54] Michael Schaarschmidt, Dominik Grewe, Dimitrios Vytiniotis, Adam Paszke, Georg Stefan Schmid, Tamara Norman, James Molloy, Jonathan Godwin, Norman Alexander Rink, Vinod Nair, et al. Automap: Towards ergonomic automated parallelism for ml models. arXiv preprint arXiv:2112.02958 , 2021. [55] Khemraj Shukla, Ameya D Jagtap, and George Em Karniadakis. Parallel physics-informed neural networks via domain decomposition. arXiv preprint arXiv:2104.10013 , 2021. [56] Dmitrii Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, and Stephan Hoyer. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences , 118(21), 2021. [57] John D Hunter. Matplotlib: A 2D graphics environment. IEEE Annals of the History of Computing , 9(03):90–95, 2007. 19 A PREPRINT - M ARCH 16, 2022 [58] Charles R Harris, K Jarrod Millman, Stéfan J van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J Smith, et al. Array programming with numpy. Nature ,585(7825):357–362, 2020. [59] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning , pages 1126–1135. PMLR, 2017. [60] Tobin A Driscoll, Nicholas Hale, and Lloyd N Trefethen. Chebfun guide, 2014. [61] Steven M Cox and Paul C Matthews. Exponential time differencing for stiff systems. Journal of Computational Physics , 176(2):430–455, 2002. 20 A PREPRINT - M ARCH 16, 2022 

## A Nomenclature 

Table 2 summarizes the main symbols and notations used in this work. Notation Description PDE Partial differential equation PINN Physics-informed neural network NTK Neural Tangent Kernel 

u(·) solution of a PDE 

N [·] a linear or non-linear differential operator 

B[·] a boundary operator 

uθ (·) neural network representation of the latent PDE solution 

θ all trainable parameters of a neural network 

Nt number of temporal collocation points 

Nx number of spatial collocation points 

wi residual weights at time ti

 causality parameter 

δ stopping criterion threshold for terminating a training loop 

Lr (t, θ) temporal residual loss 

L(θ) aggregate training loss Table 2: Nomenclature : Summary of the main symbols and notations used in this work. 21 A PREPRINT - M ARCH 16, 2022 

## B Hyper-Parameters 

Table 3 summarizes the network hyper-parameters for all numerical experiments. We tuned these hyper-parameters manually, without attempting to find the absolute best hyper-parameter setting. This process can be automated in the future leveraging effective techniques for meta-learning and hyper-parameter optimization [59]. Case Architecture Depth Width Nt Nx

Allen-Cahn MLP 6 128 100 256 Modified MLP 6 128 100 256 Lorentz MLP 5 512 256 -Kuramoto–Sivashinsky (regular) Modified MLP 5 256 32 64 Kuramoto–Sivashinsky (chaotic) Modified MLP 10 128 32 256 Navier-Stokes Modified MLP 6 128 64 512 Table 3: Network architectures for each benchmark employed in this work. 

## C Computational Cost 

Training: Table 4 summarizes the computational cost of training PINNs. The size of different models as well as network architectures are listed table 3. All networks are trained using NVIDIA RTX A6000 graphics cards. Case Architecture # Time windows Max. Iterations Training time (iter/sec) Allen-Cahn MLP 1 3 × 10 5 120.30 Modified MLP 1 3 × 10 5 58.42 Lorentz MLP 40 1 × 10 5 957.41 Kuramoto–Sivashinsky (regular) Modified MLP 10 2 × 10 5 164.77 Kuramoto–Sivashinsky (chaotic) Modified MLP 5 2 × 10 5 28.22 Navier-Stokes Modified MLP 10 1 × 10 5 68.29 Table 4: Computational cost reported timings are obtained on NVIDIA RTX A6000 graphics cards. We remark that "Max Iteration" is the maximum iteration for every tolerance  in each time window. The default tolerance list is 

[10 −2, 10 −1, 10 0, 10 1, 10 2] unless otherwise stated. The total number of iterations may vary for different examples due to the stopping criterion (see Algorithm 1). 22 A PREPRINT - M ARCH 16, 2022 

## D Allen-Cahn equation 

Validation: We solve the Allen-Cahn equation using conventional spectral methods. Specifically, assuming periodic boundary conditions, we start from the initial condition u0(x) = x2 cos( πx ) and integrate the system up to the final time 

T = 1 . Synthetic validation data are generated using the Chebfun package [ 60 ] with a spectral Fourier discretization with 512 modes and a fourth-order stiff time-stepping scheme (ETDRK4) [61] with time-step size 10 −5.−1 0 1

x

> −1.0
> −0.50.0
> u(t, x )

t = 0   

> −101

x

> −1.0
> −0.50.0
> u(t, x )

t = 0 .5  

> −101

x

> −101
> u(t, x )

t = 1 .0

Figure 15: Allen-Cahn equation: Top: Exact solution versus the prediction of a trained physics-informed neural network using Algorithm 1 and modified MLP. The resulting relative L2 error is 2.46 e − 04 . Bottom: Comparison of the predicted and exact solutions corresponding to the three temporal snapshots at t = 0 .0, 0.5, 1.0.0 1 2 3                 

> Iteration ×10 5
> 10 −7
> 10 −5
> 10 −3
> 10 −1
> 10 1
> Loss
> Lic
> Lr
> 0.00 0.25 0.50 0.75 1.00
> t
> 10 −6
> 10 −4
> 10 −2
> 10 0
> L(t, θ )
> 0.00 0.25 0.50 0.75 1.00
> t
> 0.00.20.40.60.81.0
> Temproal weights  w
> Iter = 0 Iter = 1 ×10 3
> Iter = 1 ×10 4
> Iter = 1 ×10 5
> Iter = 3 ×10 5

Figure 16: Allen-Cahn equation: Left: Loss convergence of training a physics-informed neural network using Algorithm 1. Middle: Temporal residual loss L(t, θ) at different training iteration. Right: Temporal weights at different training iteration. 23 A PREPRINT - M ARCH 16, 2022 

## E Lorentz system 

Validation: The reference solution is obtained using scipy.integrate.odeint with default settings. 

PINNs implementation: We split the whole domain [0 , 20] into 40 disjoint time windows of size ∆t = 0 .5. For each time window, we proceed by representing the latent variables of interest by a 5-layer fully-connected neural network 

uθ with 512 neurons per hidden layer 

t uθ

−−→ [xθ , y θ , z θ ]. (E.1) Since Lorentz system is highly sensitive to the initial condition, we exactly impose the initial condition by 

ˆxθ (t) = xθ (t) · t + x(0) , (E.2) 

ˆyθ (t) = yθ (t) · t + y(0) , (E.3) 

ˆzθ (t) = zθ (t) · t + z(0) . (E.4) Then the loss function can be reduced to the residual loss 

Lr (θ) = 1

NtNt∑

> i=1

wi

∣∣∣∣

dˆ xθ

dt (ti) − σ (ˆ yθ (ti) − ˆxθ (ti)) 

∣∣∣∣ (E.5) 

+ 1

NtNt∑

> i=1

wi

∣∣∣∣

dˆ yθ

dt (ti) − ˆxθ (ti)( ρ − ˆzθ (ti)) − ˆyθ (ti)

∣∣∣∣ (E.6) 

+ 1

NtNt∑

> i=1

wi

∣∣∣∣

dˆ zθ

dt (ti) − ˆxθ (ti)ˆ yθ (ti) + β ˆzθ (ti)

∣∣∣∣ , (E.7) where {ti}Nt 

> i=1

is a uniform grid in [0 , ∆t]. For this example, we set Nt = 256 and train the network with full-batch gradient descent. The temporal weights are updated by the proposed algorithm. 24 A PREPRINT - M ARCH 16, 2022 0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [0 .0, 0.5] 

t ∈ [0 .5, 1.0] 

t ∈ [1 .0, 1.5] 

t ∈ [1 .5, 2.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [2 .0, 2.5] 

t ∈ [2 .5, 3.0] 

t ∈ [3 .0, 3.5] 

t ∈ [3 .5, 4.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [4 .0, 4.5] 

t ∈ [4 .5, 5.0] 

t ∈ [5 .0, 5.5] 

t ∈ [5 .5, 6.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [6 .0, 6.5] 

t ∈ [6 .5, 7.0] 

t ∈ [7 .0, 7.5] 

t ∈ [7 .5, 8.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [8 .0, 8.5] 

t ∈ [8 .5, 9.0] 

t ∈ [9 .0, 9.5] 

t ∈ [9 .5, 10 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [10 .0, 10 .5] 

t ∈ [10 .5, 11 .0] 

t ∈ [11 .0, 11 .5] 

t ∈ [11 .5, 12 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [12 .0, 12 .5] 

t ∈ [12 .5, 13 .0] 

t ∈ [13 .0, 13 .5] 

t ∈ [13 .5, 14 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [14 .0, 14 .5] 

t ∈ [14 .5, 15 .0] 

t ∈ [15 .0, 15 .5] 

t ∈ [15 .5, 16 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [16 .0, 16 .5] 

t ∈ [16 .5, 17 .0] 

t ∈ [17 .0, 17 .5] 

t ∈ [17 .5, 18 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [18 .0, 18 .5] 

t ∈ [18 .5, 19 .0] 

t ∈ [19 .0, 19 .5] 

t ∈ [19 .5, 20 .0] 

Figure 17: Lorentz system: Left: Loss convergence of training a physics-informed neural network using Algorithm 1 for every time window. 25 A PREPRINT - M ARCH 16, 2022 0 5 10 15 20 

t

> −10 010 20

Predicted x(t)    

> 0510 15 20

t

> −20
> −10 010 20

Predicted y(t)    

> 0510 15 20

t 

> 010 20 30 40 50

Predicted z(t)    

> 0510 15 20

t

> 10 −4
> 10 −2
> 10 0

Absolute error x(t)    

> 0510 15 20

t

> 10 −4
> 10 −2
> 10 0

Absolute error y(t)    

> 0510 15 20

t

> 10 −4
> 10 −2
> 10 0

Absolute error z(t)

Figure 18: Lorentz system: Reference solutions versus the predicted solutions obtained by training a physics-informed neural network using Algorithm 1 with fixed iterations. 26 A PREPRINT - M ARCH 16, 2022 0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [0 .0, 0.5] 

t ∈ [0 .5, 1.0] 

t ∈ [1 .0, 1.5] 

t ∈ [1 .5, 2.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [2 .0, 2.5] 

t ∈ [2 .5, 3.0] 

t ∈ [3 .0, 3.5] 

t ∈ [3 .5, 4.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [4 .0, 4.5] 

t ∈ [4 .5, 5.0] 

t ∈ [5 .0, 5.5] 

t ∈ [5 .5, 6.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [6 .0, 6.5] 

t ∈ [6 .5, 7.0] 

t ∈ [7 .0, 7.5] 

t ∈ [7 .5, 8.0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [8 .0, 8.5] 

t ∈ [8 .5, 9.0] 

t ∈ [9 .0, 9.5] 

t ∈ [9 .5, 10 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [10 .0, 10 .5] 

t ∈ [10 .5, 11 .0] 

t ∈ [11 .0, 11 .5] 

t ∈ [11 .5, 12 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [12 .0, 12 .5] 

t ∈ [12 .5, 13 .0] 

t ∈ [13 .0, 13 .5] 

t ∈ [13 .5, 14 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [14 .0, 14 .5] 

t ∈ [14 .5, 15 .0] 

t ∈ [15 .0, 15 .5] 

t ∈ [15 .5, 16 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [16 .0, 16 .5] 

t ∈ [16 .5, 17 .0] 

t ∈ [17 .0, 17 .5] 

t ∈ [17 .5, 18 .0] 

0 1 2 3 4 5

Iteration ×10 5

10 −5

10 −3

10 −1

10 1

> Lr

t ∈ [18 .0, 18 .5] 

t ∈ [18 .5, 19 .0] 

t ∈ [19 .0, 19 .5] 

t ∈ [19 .5, 20 .0] 

Figure 19: Lorentz system: Left: Loss convergence of training a physics-informed neural network using Algorithm 1 for every time window. 27 A PREPRINT - M ARCH 16, 2022 

## F Kuramoto–Sivashinsky equation 

Validation: For case I (regular), we solve the Kuramoto–Sivashinsky equation using conventional spectral methods. Specifically, assuming periodic boundary conditions, we start from the initial condition u0(x) = − sin( πx ) and integrate the Equation 5.4 up to the final time T = 1 . Synthetic validation data are generated using the Chebfun package [ 60 ] with a spectral Fourier discretization with 512 modes and a fourth-order stiff time-stepping scheme (ETDRK4) [ 61 ] with time-step size 10 −5. For case II (chaotic), we perform the same procedure with the initial condition 

u0(x) = cos( x)(1 + sin( x)) . Then we select the numerical solution at t = 0 .5 as our initial condition for the PINNs simulation. 

PINNs implementation: For Case I (regular), we split the temporal domain [0 , 1] into 10 time windows of size 

∆t = 0 .1. Then we approximate the solution of each time window by a 5-layer modified MLP uθ with 256 neurons per hidden layer and encoded periodicity. It allows us to define the PDE residual by 

R[uθ ] = ∂u θ

∂t + αu θ

∂u θ

∂x + β ∂2uθ

∂x 2 + γ ∂4uθ

∂x 4 . (F.1) Then, we can formulate the following loss function 

L(θ) = 1

NtNt∑

> i=0

wiL(ti, θ), (F.2) where 

L(t0, θ) = λic 

1

NxNx∑

> j=1

|uθ (0 , x j ) − u0(xj )|2 , (F.3) 

L(ti, θ) = 1

NxNx∑

> j=1

|R [uθ ]( ti, x j )|2 , for 1 ≤ i ≤ Nt. (F.4) Here we set Nt = 32 , N x = 64 and {ti}Nt

> i=1

, {xj }Nx 

> j=1

are randomly sampled in [0 , ∆t] and [−1, 1] , respectively at each iteration of gradient descent. Particularly, we take λic = 10 3 for better enforcing the initial condition. The network is trained by minimizing the above loss function via mini-batch gradient descent using the proposed algorithm. For Case II (chaotic): We split the temporal domain [0 , 0.5] into 5 time windows of size ∆t = 0 .1. Then we perform the same procedure except for employing a 10-layer modified MLP with 128 neurons per hidden layer and setting 

λic = 10 4.

Remark: For both cases, we employ Taylor-mode automatic differentiation [ 47 ] to accelerate the computation of high-order derivatives (see section 4). 28 A PREPRINT - M ARCH 16, 2022 0 2 4

Iteration ×10 5

10 −7

10 −4

10 −1

> Lic

0 2 4

Iteration ×10 5

10 −6

10 −3

10 0

10 3               

> Lr
> t∈[0 .0,0.1]
> t∈[0 .1,0.2]
> t∈[0 .2,0.3]
> t∈[0 .3,0.4]
> t∈[0 .4,0.5]

0 2 4

Iteration ×10 5

10 −7

10 −4

10 −1

> Lic

0 2 4

Iteration ×10 5

10 −6

10 −3

10 0

10 3               

> Lr
> t∈[0 .5,0.6]
> t∈[0 .6,0.7]
> t∈[0 .7,0.8]
> t∈[0 .8,0.9]
> t∈[0 .9,1.0]

Figure 20: Kuramoto–Sivashinsky equation (regular): Loss convergence of training a physics-informed neural network using Algorithm 1 for every time window. 0 2 4 6 8

Iteration ×10 5

10 −7

10 −4

10 −1

> Lic

0 2 4 6 8

Iteration ×10 5

10 −5

10 −2

10 1               

> Lr
> t∈[0 .0,0.1]
> t∈[0 .1,0.2]
> t∈[0 .2,0.3]
> t∈[0 .3,0.4]
> t∈[0 .4,0.5]

Figure 21: Kuramoto–Sivashinsky equation (chaotic): Loss convergence of training a physics-informed neural network using Algorithm 1 for every time window. 29 A PREPRINT - M ARCH 16, 2022 

## G Navier-Stokes equation 

Validation: We simulate two-dimensional decaying turbulence in a periodic box using the JAX-CFD [ 56 ] incom-pressible Navier-Stokes solver. A high-resolution validation data-set is created by simulating an initial divergence free velocity field with the given maximum velocity vmax = 5 . The flow is solved using a Fourier spectral collocation method on a 1024 × 1024 uniform mesh with a time step of dt = 10 −4 [56]. 

PINNs implementation: Similar to the previous examples, the time domain [0 , 1] is decomposed into 10 time windows of size ∆t = 0 .1. We proceed by representing the velocity field by a 6-layer modified MLP with 128 neurons per hidden layer 

[t, x, y ] uθ

−−→ [uθ , v θ ]. (G.1) Then the vorticity can be approximated by wθ = ∂xvθ − ∂y uθ using automatic differentiation. Now we can define the PDE residual by 

Rw 

> θ

= ∂w θ

∂t + uθ

∂w θ

∂x + vθ

∂w θ

∂y − 1

Re ( ∂2wθ

∂x 2 + ∂2wθ

∂y 2 ), (G.2) 

Rc 

> θ

= ∂u θ

∂x + ∂v θ

∂y . (G.3) It allows to define the loss function by 

L(θ) = 1

NtNt∑

> i=0

wiL(ti, θ), (G.4) where 

L(t0, θ) = λic 

NxNx∑

> j=1

|uθ (0 , x j , y j ) − u0(0 , x j , y j )|2 (G.5) 

+ |vθ (0 , x j , y j ) − v0(0 , x j , y j )|2 (G.6) 

+ |wθ (0 , x j , y j ) − w0(0 , x j , y j )|2 (G.7) and 

L(ti, θ) = λw

NxNx∑

> j=1

|R w 

> θ

(ti, x j , y j )|2 + λc

NxNx∑

> j=1

|R c 

> θ

(ti, x j , y j )|2 , for 1 ≤ i ≤ Nt. (G.8) For this example we set Nt = 64 , N x = 512 and λw = 1 , λ c = 10 2, λ ic = 10 4. The temporal and spatial collocation points are randomly sampled from [0 , 1] and [0 , 2π]2, respectively. It is worth noting that we also enforce the initial velocity field (u0, v 0) as additional constraints for better convergence. This is not a severe restriction since the velocity field can be obtained from the vorticity by solving the associated Poisson’s equation or from the network representation directly. Furthermore, in Appendix we also present our results simulating the turbulent flow up to T = 2 . Figure 26 presents the visualizations of the predicted velocity and vorticity field at the final state. The predictive accuracy is quantified in Figure 27. Although the resulting relative L2 error is above 10% , our model predictions seem to be qualitatively correct against the corresponding ground truth. 30 A PREPRINT - M ARCH 16, 2022 

Figure 22: Navier-Stokes: Representative snapshots of the predicted u against the ground truth at t = 0 .2, 0.4, 0.6, 0.8.31 A PREPRINT - M ARCH 16, 2022 

Figure 23: Navier-Stokes: Representative snapshots of the predicted v against the ground truth at t = 0 .2, 0.4, 0.6, 0.8.32 A PREPRINT - M ARCH 16, 2022 

Figure 24: Navier-Stokes: Representative snapshots of the predicted w against the ground truth at t = 0 .2, 0.4, 0.6, 0.8.33 A PREPRINT - M ARCH 16, 2022 0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .0, 0.1] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .1, 0.2] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .2, 0.3] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .3, 0.4] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .4, 0.5] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .5, 0.6] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .6, 0.7] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .7, 0.8] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .8, 0.9] 

0 1 2 3

Iteration ×10 5

10 −5

10 −2

10 1

> Loss

t ∈ [0 .9, 1.0]     

> Lu0Lv0Lw0Lrw Lrc

Figure 25: Navier-Stokes: Loss convergence of training a physics-informed neural network using Algorithm 1 for every time window. 34 A PREPRINT - M ARCH 16, 2022 

Figure 26: Navier-Stokes: Predicted u, v, w against the ground truth at t = 2 .0 1 2

t

> 10 −3
> 10 −2
> 10 −1
> Rel.  L2 error

u(t, x, y )  

> 012

t

> 10 −3
> 10 −2
> 10 −1
> Rel.  L2 error

v(t, x, y )  

> 012

t

> 10 −3
> 10 −2
> 10 −1
> Rel.  L2 error

w(t, x, y )

Figure 27: Navier-Stokes: Relative L2 errors of u, v, w , respectively. 35
