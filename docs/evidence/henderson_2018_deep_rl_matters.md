Source: https://arxiv.org/abs/1709.06560
Title: Deep Reinforcement Learning that Matters - Henderson et al. (2018)
Fetched-via: curl https://r.jina.ai/https://arxiv.org/pdf/1709.06560
Fetch-status: verbatim

Title: Deep Reinforcement Learning that Matters

URL Source: https://arxiv.org/pdf/1709.06560

Published Time: Sun, 22 Jan 2023 19:19:13 GMT

Number of Pages: 26

Markdown Content:
# Deep Reinforcement Learning that Matters 

## Peter Henderson 1∗, Riashat Islam 1,2 ∗, Philip Bachman 2

## Joelle Pineau 1, Doina Precup 1, David Meger 1 

> 1

McGill University, Montreal, Canada  

> 2

Microsoft Maluuba, Montreal, Canada 

{peter.henderson,riashat.islam}@mail.mcgill.ca , phbachma@microsoft.com {jpineau,dprecup}@cs.mcgill.ca , dmeger@cim.mcgill.ca 

Abstract 

In recent years, significant progress has been made in solving challenging problems across various domains using deep re-inforcement learning (RL). Reproducing existing work and accurately judging the improvements offered by novel meth-ods is vital to sustaining this progress. Unfortunately, repro-ducing results for state-of-the-art deep RL methods is seldom straightforward. In particular, non-determinism in standard benchmark environments, combined with variance intrinsic to the methods, can make reported results tough to interpret. Without significance metrics and tighter standardization of experimental reporting, it is difficult to determine whether im-provements over the prior state-of-the-art are meaningful. In this paper, we investigate challenges posed by reproducibility, proper experimental techniques, and reporting procedures. We illustrate the variability in reported metrics and results when comparing against common baselines and suggest guidelines to make future results in deep RL more reproducible. We aim to spur discussion about how to ensure continued progress in the field by minimizing wasted effort stemming from results that are non-reproducible and easily misinterpreted. 

## Introduction 

Reinforcement learning (RL) is the study of how an agent can interact with its environment to learn a policy which maximizes expected cumulative rewards for a task. Recently, RL has experienced dramatic growth in attention and interest due to promising results in areas like: controlling continuous systems in robotics (Lillicrap et al . 2015a), playing Go (Silver et al . 2016), Atari (Mnih et al . 2013), and competitive video games (Vinyals et al . 2017; Silva and Chaimowicz 2017). Figure 1 illustrates growth of the field through the number of publications per year. To maintain rapid progress in RL research, it is important that existing works can be easily reproduced and compared to accurately judge improvements offered by novel methods. However, reproducing deep RL results is seldom straight-forward, and the literature reports a wide range of results for the same baseline algorithms (Islam et al . 2017). Re-producibility can be affected by extrinsic factors (e.g. hy-perparameters or codebases) and intrinsic factors (e.g. ef-

> ∗

These two authors contributed equally Copyright c© 2018, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved. 

Figure 1: Growth of published reinforcement learning papers. Shown are the number of RL-related publications (y-axis) per year (x-axis) scraped from Google Scholar searches. fects of random seeds or environment properties). We inves-tigate these sources of variance in reported results through a representative set of experiments. For clarity, we focus our investigation on policy gradient (PG) methods in con-tinuous control. Policy gradient methods with neural net-work function approximators have been particularly suc-cessful in continuous control (Schulman et al . 2015a; 2017; Lillicrap et al . 2015b) and are competitive with value-based methods in discrete settings. We note that the diversity of metrics and lack of significance testing in the RL literature creates the potential for misleading reporting of results. We demonstrate possible benefits of significance testing using techniques common in machine learning and statistics. Several works touch upon evaluating RL algorithms. Duan et al . (2016) benchmark several RL algorithms and provide the community with baseline implementations. Generaliz-able RL evaluation metrics are proposed in (Whiteson et al .2011). Machado et al . (2017) revisit the Arcade Learning Environment to propose better evaluation methods in these benchmarks. However, while the question of reproducibility and good experimental practice has been examined in related fields (Wagstaff 2012; Boulesteix, Lauer, and Eugster 2013; Stodden, Leisch, and Peng 2014; Bouckaert and Frank 2004; Bouckaert 2004; Vaughan and Wawerla 2012), to the best of our knowledge this is the first work to address this important question in the context of deep RL. In each section of our experimental analysis, we pose ques-tions regarding key factors affecting reproducibility. We find that there are numerous sources of non-determinism when reproducing and comparing RL algorithms. To this end, we show that fine details of experimental procedure can be crit-

> arXiv:1709.06560v3 [cs.LG] 30 Jan 2019

ical. Based on our experiments, we conclude with possible recommendations, lines of investigation, and points of dis-cussion for future works to ensure that deep reinforcement learning is reproducible and continues to matter. 

## Technical Background 

This work focuses on several model-free policy gradient algorithms with publicly available implementations which appear frequently in the literature as baselines for compar-ison against novel methods. We experiment with Trust Re-gion Policy Optimization (TRPO) (Schulman et al . 2015a), Deep Deterministic Policy Gradients (DDPG) (Lillicrap et al . 2015b), Proximal Policy Optimization (PPO) (Schulman et al . 2017), and Actor Critic using Kronecker-Factored Trust Region (ACKTR) (Wu et al . 2017). These methods have shown promising results in continuous control MuJoCo domain tasks (Todorov, Erez, and Tassa 2012) from Ope-nAI Gym (Brockman et al . 2016). Generally, they optimize 

ρ(θ, s 0) = Eπθ [∑∞ 

> t=0

γtr(st)|s0], using the policy gradient theorem: δρ (θ,s 0) 

> δθ

= ∑ 

> s

μπθ (s|s0) ∑  

> aδπ θ(a|s)
> δθ

Qπθ (s, a ).Here, μπθ (s|s0) = ∑∞ 

> t=0

γtP (st = s|s0) (Sutton et al .2000). TRPO (Schulman et al . 2015a) and PPO (Schulman et al . 2017) use constraints and advantage estimation to per-form this update, reformulating the optimization problem as: max θ Et

[ πθ (at|st)  

> πθold (at|st)

At(st, a t)

]

. Here, At is the general-ized advantage function (Schulman et al . 2015b). TRPO uses conjugate gradient descent as the optimization method with a KL constraint: Et [KL [πθold (·| st), π θ (·| st)]] ≤ δ. PPO re-formulates the constraint as a penalty (or clipping objective). DDPG and ACKTR use actor-critic methods which estimate 

Q(s, a ) and optimize a policy that maximizes the Q-function based on Monte-Carlo rollouts. DDPG does this using deter-ministic policies, while ACKTR uses Kronecketer-factored trust regions to ensure stability with stochastic policies. 

## Experimental Analysis 

We pose several questions about the factors affecting repro-ducibility of state-of-the-art RL methods. We perform a set of experiments designed to provide insight into the questions posed. In particular, we investigate the effects of: specific hyperparameters on algorithm performance if not properly tuned; random seeds and the number of averaged experi-ment trials; specific environment characteristics; differences in algorithm performance due to stochastic environments; differences due to codebases with most other factors held constant. For most of our experiments 1, except for those com-paring codebases, we generally use the OpenAI Baselines 2

implementations of the following algorithms: ACKTR (Wu et al . 2017), PPO (Schulman et al . 2017), DDPG (Plappert et al . 2017), TRPO (Schulman et al . 2017). We use the Hopper-v1 and HalfCheetah-v1 MuJoCo (Todorov, Erez, and Tassa 2012) environments from OpenAI Gym (Brockman et al .2016). These two environments provide contrasting dynam-ics (the former being more unstable).  

> 1Specific details can be found in the supplemental and code can be found at: https://git.io/vFHnf
> 2https://www.github.com/openai/baselines

To ensure fairness we run five experiment trials for each evaluation, each with a different preset random seed (all experiments use the same set of random seeds). In all cases, we highlight important results here, with full descriptions of experimental setups and additional learning curves included in the supplemental material. Unless otherwise mentioned, we use default settings whenever possible, while modifying only the hyperparameters of interest. All results (including graphs) show mean and standard error across random seeds. We use multilayer perceptron function approximators in all cases. We denote the hidden layer sizes and activations as (N, M, activation ). For default settings, we vary the hy-perparameters under investigation one at a time. For DDPG we use a network structure of (64 , 64 , ReLU ) for both actor and critic. For TRPO and PPO, we use (64 , 64 , tanh ) for the policy. For ACKTR, we use (64 , 64 , tanh ) for the actor and 

(64 , 64 , ELU ) for the critic. 

Hyperparameters 

What is the magnitude of the effect hyperparameter settings can have on baseline performance? 

Tuned hyperparameters play a large role in eliciting the best results from many algorithms. However, the choice of op-timal hyperparameter configuration is often not consistent in related literature, and the range of values considered is often not reported 3. Furthermore, poor hyperparameter selec-tion can be detrimental to a fair comparison against baseline algorithms. Here, we investigate several aspects of hyperpa-rameter selection on performance. 

Network Architecture 

How does the choice of network architecture for the policy and value function approximation affect performance? 

In (Islam et al . 2017), it is shown that policy network architec-ture can significantly impact results in both TRPO and DDPG. Furthermore, certain activation functions such as Rectified Linear Unit (ReLU) have been shown to cause worsened learning performance due to the “dying relu” problem (Xu et al . 2015). As such, we examine network architecture and ac-tivation functions for both policy and value function approxi-mators. In the literature, similar lines of investigation have shown the differences in performance when comparing linear approximators, RBFs, and neural networks (Rajeswaran et al . 2017). Tables 1 and 2 summarize the final evaluation per-formance of all architectural variations after training on 2M samples (i.e. 2M timesteps in the environment). All learning curves and details on setup can be found in the supplemental material. We vary hyperparameters one at a time, while using a default setting for all others. We investigate three multilayer perceptron (MLP) architectures commonly seen in the liter-ature: (64 , 64) , (100 , 50 , 25) , and (400 , 300) . Furthermore, we vary the activation functions of both the value and policy networks across tanh, ReLU, and Leaky ReLU activations. 

Results Figure 2 shows how significantly performance can be affected by simple changes to the policy or value network                   

> 3A sampled literature review can be found in the supplemental. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
> Timesteps ×10 6
> −2000
> −1000
> 0
> 1000
> 2000
> Average Return
> HalfCheetah-v1 (PPO, Policy Network Structure)
> (64,64)
> (100,50,25)
> (400,300) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
> Timesteps ×10 6
> −750
> −500
> −250
> 0
> 250
> 500
> 750
> 1000
> Average Return
> HalfCheetah-v1 (TRPO, Policy Network Activation)
> tanh
> relu
> leaky relu

Figure 2: Significance of Policy Network Structure and Activation Functions PPO (left), TRPO (middle) and DDPG (right). 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00           

> Timesteps ×10 6
> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return
> HalfCheetah-v1 (DDPG, Reward Scale, Layer Norm)
> rs=1e-4
> rs=1e-3
> rs=1e-2
> rs=1e-1
> rs=1
> rs=10
> rs=100 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
> Timesteps ×10 6
> 0
> 1000
> 2000
> 3000
> Average Return
> HalfCheetah-v1 (DDPG, Reward Scale, No Layer Norm)
> rs=1e-4
> rs=1e-3
> rs=1e-2
> rs=1e-1
> rs=1
> rs=10
> rs=100

Figure 3: DDPG reward rescaling on HalfCheetah-v1, with and without layer norm. activations. We find that usually ReLU or Leaky ReLU acti-vations perform the best across environments and algorithms. The effects are not consistent across algorithms or environ-ments. This inconsistency demonstrates how interconnected network architecture is to algorithm methodology. For exam-ple, using a large network with PPO may require tweaking other hyperparameters such as the trust region clipping or learning rate to compensate for the architectural change 4.This intricate interplay of hyperparameters is one of the rea-sons reproducing current policy gradient methods is so dif-ficult. It is exceedingly important to choose an appropriate architecture for proper baseline results. This also suggests a possible need for hyperparameter agnostic algorithms—that is algorithms that incorporate hyperparameter adaptation as part of the design—such that fair comparisons can be made without concern about improper settings for the task at hand. 

Reward Scale 

How can the reward scale affect results? Why is reward rescaling used? 

Reward rescaling has been used in several recent works (Duan et al . 2016; Gu et al . 2016) to improve results for DDPG. This involves simply multiplying the rewards gen-erated from an environment by some scalar ( ˆr = rˆσ) for training. Often, these works report using a reward scale of ˆσ = 0 .1. In Atari domains, this is akin to clipping the rewards to [0 , 1] . By intuition, in gradient based methods (as used in most deep RL) a large and sparse output scale can result in problems regarding saturation and inefficiency in learning (LeCun et al . 2012; Glorot and Bengio 2010; Vincent, de Br ´ebisson, and Bouthillier 2015). Therefore clip-ping or rescaling rewards compresses the space of estimated 

> 4

We find that the KL divergence of updates with the large net-work (400 , 300) seen in Figure 2 is on average 33 .52 times higher than the KL divergence of updates with the (64 , 64) network. 

expected returns in action value function based methods such as DDPG. We run a set of experiments using reward rescaling in DDPG (with and without layer normalization) for insights into how this aspect affects performance. 

Results Our analysis shows that reward rescaling can have a large effect (full experiment results can be found in the supplemental material), but results were inconsistent across environments and scaling values. Figure 3 shows one such ex-ample where reward rescaling affects results, causing a failure to learn in small settings below ˆσ = 0 .01 . In particular, layer normalization changes how the rescaling factor affects results, suggesting that these impacts are due to the use of deep net-works and gradient-based methods. With the value function approximator tracking a moving target distribution, this can potentially affect learning in unstable environments where a deep Q-value function approximator is used. Furthermore, some environments may have untuned reward scales (e.g. the HumanoidStandup-v1 of OpenAI gym which can reach rewards in the scale of millions). Therefore, we suggest that this hyperparameter has the potential to have a large impact if considered properly. Rather than rescaling rewards in some environments, a more principled approach should be taken to address this. An initial foray into this problem is made in (van Hasselt et al . 2016), where the authors adaptively rescale reward targets with normalized stochastic gradient, but further research is needed. 

Random Seeds and Trials 

Can random seeds drastically alter performance? Can one distort results by averaging an improper number of trials? 

A major concern with deep RL is the variance in results due to environment stochasticity or stochasticity in the learning process (e.g. random weight initialization). As such, even averaging several learning results together across totally dif-ferent random seeds can lead to the reporting of misleading results. We highlight this in the form of an experiment. Algorithm Environment 400,300 64,64 100,50,25 tanh ReLU LeakyReLU 

TRPO Hopper-v1 2980 ± 35 2674 ± 227 3110 ± 78 2674 ± 227 2772 ± 211 -(Schulman et al. 2015a) HalfCheetah-v1 1791 ± 224 1939 ± 140 2151 ± 27 1939 ± 140 3041 ± 161 -TRPO Hopper-v1 1243 ± 55 1303 ± 89 1243 ± 55 1303 ± 89 1131 ± 65 1341 ± 127 (Duan et al. 2016) HalfCheetah-v1 738 ± 240 834 ± 317 850 ±378 834 ± 317 784 ± 352 1139 ±364 TRPO Hopper-v1 2909 ± 87 2828 ± 70 2812 ± 88 2828 ± 70 2941 ± 91 2865 ± 189 (Schulman et al. 2017) HalfCheetah-v1 -155 ± 188 205 ± 256 306 ± 261 205 ± 256 1045 ± 114 778 ± 177 PPO Hopper-v1 61 ± 33 2790 ± 62 2592 ± 196 2790 ± 62 2695 ± 86 2587 ± 53 (Schulman et al. 2017) HalfCheetah-v1 -1180 ± 444 2201 ± 323 1314 ± 340 2201 ± 323 2971 ± 364 2895 ± 365 DDPG Hopper-v1 1419 ± 313 1632 ± 459 2142 ± 436 1491 ± 205 1632 ± 459 1384 ± 285 (Plappert et al. 2017) HalfCheetah-v1 5579 ± 354 4198 ± 606 5600 ± 601 5325 ± 281 4198 ± 606 4094 ± 233 DDPG Hopper-v1 600 ± 126 593 ± 155 501 ± 129 436 ± 48 593 ± 155 319 ± 127 (Gu et al. 2016) HalfCheetah-v1 2845 ± 589 2771 ± 535 1638 ± 624 1638 ± 624 2771 ± 535 1405 ± 511 DDPG Hopper-v1 506 ± 208 749 ± 271 629 ± 138 354 ± 91 749 ± 271 -(Duan et al. 2016) HalfCheetah-v1 850 ± 41 1573 ± 385 1224 ± 553 1311 ± 271 1573 ± 385 -ACKTR Hopper-v1 2577 ± 529 1608 ± 66 2287 ± 946 1608 ± 66 2835 ± 503 2718 ± 434 (Wu et al. 2017) HalfCheetah-v1 2653 ± 408 2691 ± 231 2498 ± 112 2621 ± 381 2160 ± 151 2691 ± 231 

Table 1: Results for our policy architecture permutations across various implementations and algorithms. Final average ±

standard error across 5 trials of returns across the last 100 trajectories after 2M training samples. For ACKTR, we use ELU 

activations instead of leaky ReLU. 

Algorithm Environment 400,300 64,64 100,50,25 tanh ReLU LeakyReLU 

TRPO Hopper-v1 3011 ± 171 2674 ± 227 2782 ± 120 2674 ± 227 3104 ± 84 -(Schulman et al. 2015a) HalfCheetah-v1 2355 ± 48 1939 ± 140 1673 ± 148 1939 ± 140 2281 ± 91 -TRPO Hopper-v1 2909 ± 87 2828 ± 70 2812 ± 88 2828 ± 70 2829 ± 76 3047 ± 68 (Schulman et al. 2017) HalfCheetah-v1 178 ± 242 205 ± 256 172 ± 257 205 ± 256 235 ± 260 325 ± 208 PPO Hopper-v1 2704 ± 37 2790 ± 62 2969 ± 111 2790 ± 62 2687 ± 144 2748 ± 77 (Schulman et al. 2017) HalfCheetah-v1 1523 ± 297 2201 ± 323 1807 ± 309 2201 ± 323 1288 ± 12 1227 ± 462 DDPG Hopper-v1 1419 ± 312 1632 ± 458 1569 ± 453 971 ± 137 852 ± 143 843 ± 160 (Plappert et al. 2017) HalfCheetah-v1 5600 ± 601 4197 ± 606 4713 ± 374 3908 ± 293 4197 ± 606 5324 ± 280 DDPG Hopper-v1 523 ± 248 343 ± 34 345 ± 44 436 ± 48 343 ± 34 -(Gu et al. 2016) HalfCheetah-v1 1373 ± 678 1717 ± 508 1868 ± 620 1128 ± 511 1717 ± 508 -DDPG Hopper-v1 1208 ± 423 394 ± 144 380 ± 65 354 ± 91 394 ± 144 -(Duan et al. 2016) HalfCheetah-v1 789 ± 91 1095 ± 139 988 ± 52 1311 ± 271 1095 ± 139 -ACKTR Hopper-v1 152 ± 47 1930 ± 185 1589 ± 225 691 ± 55 500 ± 379 1930 ± 185 (Wu et al. 2017) HalfCheetah-v1 518 ± 632 3018 ± 386 2554 ± 219 2547 ± 172 3362 ± 682 3018 ± 38 

Table 2: Results for our value function ( Q or V ) architecture permutations across various implementations and algorithms. Final average ± standard error across 5 trials of returns across the last 100 trajectories after 2M training samples. For ACKTR, we use 

ELU activations instead of leaky ReLU. 

Figure 4: Performance of several policy gradient algorithms across benchmark MuJoCo environment suites 

Environment DDPG ACKTR TRPO PPO 

HalfCheetah-v1 5037 (3664, 6574) 3888 (2288, 5131) 1254.5 (999, 1464) 3043 (1920, 4165) Hopper-v1 1632 (607, 2370) 2546 (1875, 3217) 2965 (2854, 3076) 2715 (2589, 2847) Walker2d-v1 1582 (901, 2174) 2285 (1246, 3235) 3072 (2957, 3183) 2926 (2514, 3361) Swimmer-v1 31 (21, 46) 50 (42, 55) 214 (141, 287) 107 (101, 118) 

Table 3: Bootstrap mean and 95% confidence bounds for a subset of environment experiments. 10k bootstrap iterations and the pivotal method were used. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00  

> Timesteps ×10 6
> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return
> HalfCheetah-v1 (TRPO, Different Random Seeds)
> Random Average (5 runs)
> Random Average (5 runs)

Figure 5: TRPO on HalfCheetah-v1 using the same hyperpa-rameter configurations averaged over two sets of 5 different random seeds each. The average 2-sample t-test across entire training distribution resulted in t = −9.0916 , p = 0 .0016 .

Results We perform 10 experiment trials, for the same hyperparameter configuration, only varying the random seed across all 10 trials. We then split the trials into two sets of 5 and average these two groupings together. As shown in Figure 5, we find that the performance of algorithms can be drastically different. We demonstrate that the variance between runs is enough to create statistically different dis-tributions just from varying random seeds. Unfortunately, in recent reported results, it is not uncommon for the top-N tri-als to be selected from among several trials (Wu et al . 2017; Mnih et al . 2016) or averaged over only small number of tri-als ( N < 5) (Gu et al . 2017; Wu et al . 2017). Our experiment with random seeds shows that this can be potentially mislead-ing. Particularly for HalfCheetah, it is possible to get learning curves that do not fall within the same distribution at all, just by averaging different runs with the same hyperparameters, but different random seeds. While there can be no specific number of trials specified as a recommendation, it is possible that power analysis methods can be used to give a general idea to this extent as we will discuss later. However, more investigation is needed to answer this open problem. 

Environments 

How do the environment properties affect variability in re-ported RL algorithm performance? 

To assess how the choice of evaluation environment can af-fect the presented results, we use our aforementioned default set of hyperparameters across our chosen testbed of algo-rithms and investigate how well each algorithm performs across an extended suite of continuous control tasks. For these experiments, we use the following environments from OpenAI Gym: Hopper-v1, HalfCheetah-v1, Swimmer-v1 and Walker2d-v1. The choice of environment often plays an im-portant role in demonstrating how well a new proposed algo-rithm performs against baselines. In continuous control tasks, often the environments have random stochasticity, shortened trajectories, or different dynamic properties. We demonstrate that, as a result of these differences, algorithm performance can vary across environments and the best performing algo-rithm across all environments is not always clear. Thus it is increasingly important to present results for a wide range of environments and not only pick those which show a novel work outperforming other methods. 

Results As shown in Figure 4, in environments with sta-ble dynamics (e.g. HalfCheetah-v1), DDPG outperforms all other algorithsm. However, as dynamics become more unsta-ble (e.g. in Hopper-v1) performance gains rapidly diminish. As DDPG is an off-policy method, exploration noise can cause sudden failures in unstable environments. Therefore, learning a proper Q-value estimation of expected returns is difficult, particularly since many exploratory paths will result in failure. Since failures in such tasks are characterized by shortened trajectories, a local optimum in this case would be simply to survive until the maximum length of the trajectory (corresponding to one thousand timesteps and similar reward due to a survival bonus in the case of Hopper-v1). As can be seen in Figure 4, DDPG with Hopper does exactly this. This is a clear example where showing only the favourable and sta-ble HalfCheetah when reporting DDPG-based experiments would be unfair. Furthermore, let us consider the Swimmer-v1 environment shown in Figure 4. Here, TRPO significantly outperforms all other algorithms. Due to the dynamics of the water-like environment, a local optimum for the system is to curl up and flail without proper swimming. However, this corresponds to a return of ∼130 . By reaching a local optimum, learning curves can indicate successful optimization of the policy over time, when in reality the returns achieved are not qualitatively representative of learning the desired behaviour, as demon-strated in video replays of the learned policy 5. Therefore, it is important to show not only returns but demonstrations of the learned policy in action. Without understanding what the evaluation returns indicate, it is possible that misleading results can be reported which in reality only optimize local optima rather than reaching the desired behaviour. 

Codebases 

Are commonly used baseline implementations comparable? 

In many cases, authors implement their own versions of base-line algorithms to compare against. We investigate the Ope-nAI baselines implementation of TRPO as used in (Schulman et al . 2017), the original TRPO code (Schulman et al . 2015a), and the rllab (Duan et al . 2016) Tensorflow implementation of TRPO. We also compare the rllab Theano (Duan et al . 2016), rllabplusplus (Gu et al . 2016), and OpenAI baselines (Plap-pert et al . 2017) implementations of DDPG. Our goal is to draw attention to the variance due to implementation details across algorithms. We run a subset of our architecture experi-ments as with the OpenAI baselines implementations using the same hyperparameters as in those experiments 6.

Results We find that implementation differences which are often not reflected in publications can have dramatic impacts on performance. This can be seen for our final evalu-ation performance after training on 2M samples in Tables 1 and 2, as well as a sample comparison in Figure 6. This                   

> 5https://youtu.be/lKpUQYjgm80
> 6Differences are discussed in the supplemental (e.g. use of dif-ferent optimizers for the value function baseline). Leaky ReLU activations are left out to narrow the experiment scope. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
> Timesteps ×10 6
> −500
> 0
> 500
> 1000
> 1500
> 2000
> Average Return
> HalfCheetah-v1 (TRPO, Codebase Comparison)
> Schulman 2015
> Schulman 2017
> Duan 2016 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00
> Timesteps ×10 6
> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return
> HalfCheetah-v1 (DDPG, Codebase Comparison)
> Duan 2016
> Gu 2016
> Plapper 2017

Figure 6: TRPO codebase comparison using our default set of hyperparameters (as used in other experiments). demonstrates the necessity that implementation details be enumerated, codebases packaged with publications, and that performance of baseline experiments in novel works matches the original baseline publication code. 

## Reporting Evaluation Metrics 

In this section we analyze some of the evaluation metrics commonly used in the reinforcement learning literature. In practice, RL algorithms are often evaluated by simply pre-senting plots or tables of average cumulative reward (average returns) and, more recently, of maximum reward achieved over a fixed number of timesteps. Due to the unstable na-ture of many of these algorithms, simply reporting the max-imum returns is typically inadequate for fair comparison; even reporting average returns can be misleading as the range of performance across seeds and trials is unknown. Alone, these may not provide a clear picture of an algorithm’s range of performance. However, when combined with confidence intervals, this may be adequate to make an informed deci-sion given a large enough number of trials. As such, we investigate using the bootstrap and significance testing as in ML (Kohavi and others 1995; Bouckaert and Frank 2004; Nadeau and Bengio 2000) to evaluate algorithm performance. 

Online View vs. Policy Optimization An important dis-tinction when reporting results is the online learning view versus the policy optimization view of RL. In the online view, an agent will optimize the returns across the entire learning process and there is not necessarily an end to the agent’s trajectory. In this view, evaluations can use the average cumu-lative rewards across the entire learning process (balancing exploration and exploitation) as in (Hofer and Gimbert 2016), or can possibly use offline evaluation as in (Mandel et al .2016). The alternate view corresponds to policy optimization, where evaluation is performed using a target policy in an of-fline manner. In the policy optimization view it is important to run evaluations across the entire length of the task trajectory with a single target policy to determine the average returns that the target can obtain. We focus on evaluation methods for the policy optimization view (with offline evaluation), but the same principles can be applied to the online view. 

Confidence Bounds The sample bootstrap has been a pop-ular method to gain insight into a population distribution from a smaller sample (Efron and Tibshirani 1994). Boot-strap methods are particularly popular for A/B testing, and we can borrow some ideas from this field. Generally a boot-strap estimator is obtained by resampling with replacement many times to generate a statistically relevant mean and con-fidence bound. Using this technique, we can gain insight into what is the 95% confidence interval of the results from our section on environments. Table 3 shows the bootstrap mean and 95% confidence bounds on our environment experiments. Confidence intervals can vary wildly between algorithms and environments. We find that TRPO and PPO are the most stable with small confidence bounds from the bootstrap. In cases where confidence bounds are exceedingly large, it may be necessary to run more trials (i.e. increase the sample size). 

Power Analysis Another method to determine if the sample size must be increased is bootstrap power analy-sis (Tuff ´ery 2011; Yuan and Hayashi 2003). If we use our sample and give it some uniform lift (for example, scaling uni-formly by 1.25), we can run many bootstrap simulations and determine what percentage of the simulations result in statis-tically significant values with the lift. If there is a small per-centage of significant values, a larger sample size is needed (more trials must be run). We do this across all environment experiment trial runs and indeed find that, in more unstable settings, the bootstrap power percentage leans towards in-significant results in the lift experiment. Conversely, in stable trials (e.g. TRPO on Hopper-v1) with a small sample size, the lift experiment shows that no more trials are needed to generate significant comparisons. These results are provided in the supplemental material. 

Significance An important factor when deciding on an RL algorithm to use is the significance of the reported gains based on a given metric. Several works have investigated the use of significance metrics to assess the reliability of reported evaluation metrics in ML. However, few works in reinforcement learning assess the significance of reported metrics. Based on our experimental results which indicate that algorithm performance can vary wildly based simply on perturbations of random seeds, it is clear that some metric is necessary for assessing the significance of algorithm perfor-mance gains and the confidence of reported metrics. While more research and investigation is needed to determine the best metrics for assessing RL algorithms, we investigate an initial set of metrics based on results from ML. In supervised learning, k-fold t-test, corrected resampled t-test, and other significance metrics have been discussed when comparing machine learning results (Bouckaert and Frank 2004; Nadeau and Bengio 2000). However, the assumptions pertaining to the underlying data with corrected metrics do not necessarily apply in RL. Further work is needed to inves-tigate proper corrected significance tests for RL. Nonetheless, we explore several significance measures which give insight into whether a novel algorithm is truly performing as the state-of-the-art. We consider the simple 2-sample t-test (sorting all final evaluation returns across N random trials with different random seeds); the Kolmogorov-Smirnov test (Wilcox 2005); and bootstrap percent differences with 95% confidence in-tervals. All calculated metrics can be found in the supple-mental. Generally, we find that the significance values match up to what is to be expected. Take, for example, comparing Walker2d-v1 performance of ACKTR vs. DDPG. ACKTR performs slightly better, but this performance is not signifi-cant due to the overlapping confidence intervals of the two: 

t = 1 .03 , p = 0 .334 , KS = 0 .40 , p = 0 .697 , bootstrapped percent difference 44.47% (-80.62%, 111.72%). 

## Discussion and Conclusion 

Through experimental methods focusing on PG methods for continuous control, we investigate problems with repro-ducibility in deep RL. We find that both intrinsic (e.g. random seeds, environment properties) and extrinsic sources (e.g. hy-perparameters, codebases) of non-determinism can contribute to difficulties in reproducing baseline algorithms. Moreover, we find that highly varied results due to intrinsic sources bolster the need for using proper significance analysis. We propose several such methods and show their value on a subset of our experiments. 

What recommendations can we draw from our experiments? 

Based on our experimental results and investigations, we can provide some general recommendations. Hyperparame-ters can have significantly different effects across algorithms and environments. Thus it is important to find the work-ing set which at least matches the original reported perfor-mance of baseline algorithms through standard hyperparame-ter searches. Similarly, new baseline algorithm implementa-tions used for comparison should match the original codebase results if available. Overall, due to the high variance across trials and random seeds of reinforcement learning algorithms, many trials must be run with different random seeds when comparing performance. Unless random seed selection is explicitly part of the algorithm, averaging multiple runs over different random seeds gives insight into the population dis-tribution of the algorithm performance on an environment. Similarly, due to these effects, it is important to perform proper significance testing to determine if the higher average returns are in fact representative of better performance. We highlight several forms of significance testing and find that they give generally expected results when taking confi-dence intervals into consideration. Furthermore, we demon-strate that bootstrapping and power analysis are possible ways to gain insight into the number of trial runs necessary to make an informed decision about the significance of algorithm per-formance gains. In general, however, the most important step to reproducibility is to report all hyperparameters, implemen-tation details, experimental setup, and evaluation methods for both baseline comparison methods and novel work. Without the publication of implementations and related details, wasted effort on reproducing state-of-the-art works will plague the community and slow down progress. 

What are possible future lines of investigation? 

Due to the significant effects of hyperparameters (partic-ularly reward scaling), another possibly important line of future investigation is in building hyperparameter agnostic algorithms. Such an approach would ensure that there is no unfairness introduced from external sources when compar-ing algorithms agnostic to parameters such as reward scale, batch size, or network structure. Furthermore, while we in-vestigate an initial set of significance metrics here, they may not be the best fit for comparing RL algorithms. Several works have begun investigating policy evaluation methods for the purposes of safe RL (Thomas and Brunskill 2016; Thomas, Theocharous, and Ghavamzadeh 2015), but further work is needed in significance testing and statistical analysis. Similar lines of investigation to (Nadeau and Bengio 2000; Bouckaert and Frank 2004) would be helpful to determine the best methods for evaluating performance gain significance. 

How can we ensure that deep RL matters? 

We discuss many different factors affecting reproducibility of RL algorithms. The sensitivity of these algorithms to changes in reward scale, environment dynamics, and random seeds can be considerable and varies between algorithms and set-tings. Since benchmark environments are proxies for real-world applications to gauge generalized algorithm perfor-mance, perhaps more emphasis should be placed on the appli-cability of RL algorithms to real-world tasks. That is, as there is often no clear winner among all benchmark environments, perhaps recommended areas of application should be demon-strated along with benchmark environment results when pre-senting a new algorithm. Maybe new methods should be answering the question: in what setting would this work be useful? This is something that is addressed for machine learn-ing in (Wagstaff 2012) and may warrant more discussion for RL. As a community, we must not only ensure reproducible results with fair comparisons, but we must also consider what are the best ways to demonstrate that RL continues to matter. 

## Acknowledgements 

We thank NSERC, CIFAR, the Open Philanthropy Project, and the AWS Cloud Credits for Research Program. 

## References      

> Bouckaert, R. R., and Frank, E. 2004. Evaluating the replicability of significance tests for comparing learning algorithms. In PAKDD ,3–12. Springer. Bouckaert, R. R. 2004. Estimating replicability of classifier learning experiments. In Proceedings of the 21st International Conference on Machine Learning (ICML) .Boulesteix, A.-L.; Lauer, S.; and Eugster, M. J. 2013. A plea for neutral comparison studies in computational sciences. PloS one
> 8(4):e61562. Brockman, G.; Cheung, V.; Pettersson, L.; Schneider, J.; Schulman, J.; Tang, J.; and Zaremba, W. 2016. OpenAI gym. arXiv preprint arXiv:1606.01540 .Duan, Y.; Chen, X.; Houthooft, R.; Schulman, J.; and Abbeel, P. 2016. Benchmarking deep reinforcement learning for continuous control. In Proceedings of the 33rd International Conference on Machine Learning (ICML) .

Efron, B., and Tibshirani, R. J. 1994. An introduction to the boot-strap . CRC press. Glorot, X., and Bengio, Y. 2010. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , 249–256. Gu, S.; Lillicrap, T.; Ghahramani, Z.; Turner, R. E.; and Levine, S. 2016. Q-prop: Sample-efficient policy gradient with an off-policy critic. arXiv preprint arXiv:1611.02247 .Gu, S.; Lillicrap, T.; Ghahramani, Z.; Turner, R. E.; Sch ¨olkopf, B.; and Levine, S. 2017. Interpolated policy gradient: Merging on-policy and off-policy gradient estimation for deep reinforcement learning. arXiv preprint arXiv:1706.00387 .Hofer, L., and Gimbert, H. 2016. Online reinforcement learning for real-time exploration in continuous state and action markov decision processes. arXiv preprint arXiv:1612.03780 .Islam, R.; Henderson, P.; Gomrokchi, M.; and Precup, D. 2017. Reproducibility of benchmarked deep reinforcement learning tasks for continuous control. ICML Reproducibility in Machine Learning Workshop .Kohavi, R., et al. 1995. A study of cross-validation and bootstrap for accuracy estimation and model selection. In IJCAI , volume 14. LeCun, Y. A.; Bottou, L.; Orr, G. B.; and M ¨uller, K.-R. 2012. Effi-cient backprop. In Neural Networks: Tricks of the Trade . Springer. Lillicrap, T. P.; Hunt, J. J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; and Wierstra, D. 2015a. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 .Lillicrap, T. P.; Hunt, J. J.; Pritzel, A.; Heess, N.; Erez, T.; Tassa, Y.; Silver, D.; and Wierstra, D. 2015b. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 .Machado, M. C.; Bellemare, M. G.; Talvitie, E.; Veness, J.; Hausknecht, M.; and Bowling, M. 2017. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. arXiv preprint arXiv:1709.06009 .Mandel, T.; Liu, Y.-E.; Brunskill, E.; and Popovic, Z. 2016. Offline Evaluation of Online Reinforcement Learning Algorithms. In AAAI .Mnih, V.; Kavukcuoglu, K.; Silver, D.; Graves, A.; Antonoglou, I.; Wierstra, D.; and Riedmiller, M. 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 .Mnih, V.; Badia, A. P.; Mirza, M.; Graves, A.; Lillicrap, T.; Harley, T.; Silver, D.; and Kavukcuoglu, K. 2016. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning , 1928–1937. Nadeau, C., and Bengio, Y. 2000. Inference for the generalization error. In Advances in neural information processing systems .Plappert, M.; Houthooft, R.; Dhariwal, P.; Sidor, S.; Chen, R.; Chen, X.; Asfour, T.; Abbeel, P.; and Andrychowicz, M. 2017. Parameter space noise for exploration. arXiv preprint arXiv:1706.01905 .Rajeswaran, A.; Lowrey, K.; Todorov, E.; and Kakade, S. 2017. Towards generalization and simplicity in continuous control. arXiv preprint arXiv:1703.02660 .Schulman, J.; Levine, S.; Abbeel, P.; Jordan, M.; and Moritz, P. 2015a. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML) .Schulman, J.; Moritz, P.; Levine, S.; Jordan, M.; and Abbeel, P. 2015b. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438 .Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 .Silva, V. d. N., and Chaimowicz, L. 2017. Moba: a new arena for game ai. arXiv preprint arXiv:1705.10443 .Silver, D.; Huang, A.; Maddison, C. J.; Guez, A.; Sifre, L.; Van Den Driessche, G.; Schrittwieser, J.; Antonoglou, I.; Panneershel-vam, V.; Lanctot, M.; et al. 2016. Mastering the game of go with deep neural networks and tree search. Nature 529(7587):484–489. Stadie, B. C.; Abbeel, P.; and Sutskever, I. 2017. Third-person imitation learning. arXiv preprint arXiv:1703.01703 .Stodden, V.; Leisch, F.; and Peng, R. D. 2014. Implementing reproducible research . CRC Press. Sutton, R. S.; McAllester, D. A.; Singh, S. P.; and Mansour, Y. 2000. Policy gradient methods for reinforcement learning with func-tion approximation. In Advances in neural information processing systems .Thomas, P., and Brunskill, E. 2016. Data-efficient off-policy policy evaluation for reinforcement learning. In International Conference on Machine Learning , 2139–2148. Thomas, P. S.; Theocharous, G.; and Ghavamzadeh, M. 2015. High-Confidence Off-Policy Evaluation. In AAAI .Todorov, E.; Erez, T.; and Tassa, Y. 2012. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Confer-ence on Intelligent Robots and Systems, IROS 2012, Vilamoura, Algarve, Portugal, October 7-12, 2012 , 5026–5033. Tuff ´ery, S. 2011. Data mining and statistics for decision making ,volume 2. Wiley Chichester. van Hasselt, H. P.; Guez, A.; Hessel, M.; Mnih, V.; and Silver, D. 2016. Learning values across many orders of magnitude. In 

Advances in Neural Information Processing Systems , 4287–4295. Vaughan, R., and Wawerla, J. 2012. Publishing identifiable exper-iment code and configuration is important, good and easy. arXiv preprint arXiv:1204.2235 .Vincent, P.; de Br ´ebisson, A.; and Bouthillier, X. 2015. Efficient exact gradient update for training deep networks with very large sparse targets. In Advances in Neural Information Processing Sys-tems , 1108–1116. Vinyals, O.; Ewalds, T.; Bartunov, S.; Georgiev, P.; Vezhnevets, A. S.; Yeo, M.; Makhzani, A.; K ¨uttler, H.; Agapiou, J.; Schrittwieser, J.; et al. 2017. Starcraft ii: A new challenge for reinforcement learning. arXiv preprint arXiv:1708.04782 .Wagstaff, K. 2012. Machine learning that matters. arXiv preprint arXiv:1206.4656 .Whiteson, S.; Tanner, B.; Taylor, M. E.; and Stone, P. 2011. Pro-tecting against evaluation overfitting in empirical reinforcement learning. In 2011 IEEE Symposium on Adaptive Dynamic Program-ming And Reinforcement Learning, ADPRL 2011, Paris, France, April 12-14, 2011 , 120–127. Wilcox, R. 2005. Kolmogorov–smirnov test. Encyclopedia of biostatistics .Wu, Y.; Mansimov, E.; Liao, S.; Grosse, R.; and Ba, J. 2017. Scal-able trust-region method for deep reinforcement learning using kronecker-factored approximation. arXiv preprint:1708.05144 .Xu, B.; Wang, N.; Chen, T.; and Li, M. 2015. Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853 .Yuan, K.-H., and Hayashi, K. 2003. Bootstrap approach to inference and power analysis based on three test statistics for covariance structure models. British Journal of Mathematical and Statistical Psychology 56(1):93–110. Supplemental Material 

In this supplemental material, we include a detailed review of experiment configurations of related work with policy gradient methods in continuous control MuJoCo (Todorov, Erez, and Tassa 2012) environment tasks from OpenAI Gym (Brockman et al . 2016). We include a detailed list of the hyperparameters and reported metrics typically used in policy gradient literature in deep RL. We also include all our experimental results, with baseline algorithms DDPG (Lillicrap et al . 2015b), TRPO (Schulman et al . 2015a), PPO (Schulman et al . 2017) and ACKTR (Wu et al . 2017)) as discussed in the paper. Our experimental results include figures with different hyperparameters (network architectures, activation functions) to highlight the differences this can have across algorithms and environments. Finally, as discussed in the paper, we include discussion of significance metrics and show how these metrics can be useful for evaluating deep RL algorithms. 

## Literature Reviews 

Hyperparameters 

In this section, we include a list of hyperparameters that are reported in related literature, as shown in figure 4. Our analysis shows that often there is no consistency in the type of network architectures and activation functions that are used in related literature. As shown in the paper and from our experimental results in later sections, we find, however, that these hyperparameters can have a significant effect in the performance of algorithms across benchmark environments typically used. 

Table 4: Evaluation Hyperparameters of baseline algorithms reported in related literature 

Related Work (Algorithm) Policy Network Policy Network Activation Value Network Value Network Activation Reward Scaling Batch Size 

DDPG 64x64 ReLU 64x64 ReLU 1.0 128 TRPO 64x64 TanH 64x64 TanH - 5k PPO 64x64 TanH 64x64 TanH - 2048 ACKTR 64x64 TanH 64x64 ELU - 2500 Q-Prop (DDPG) 100x50x25 TanH 100x100 ReLU 0.1 64 Q-Prop (TRPO) 100x50x25 TanH 100x100 ReLU - 5k IPG (TRPO) 100x50x25 TanH 100x100 ReLU - 10k Param Noise (DDPG) 64x64 ReLU 64x64 ReLU - 128 Param Noise (TRPO) 64x64 TanH 64x64 TanH - 5k Benchmarking (DDPG) 400x300 ReLU 400x300 ReLU 0.1 64 Benchmarking (TRPO) 100x50x25 TanH 100x50x25 TanH - 25k 

Reported Results on Benchmarked Environments 

We then demonstrate how experimental reported results, on two different environments (HalfCheetah-v1 and Hopper-v1) can vary across different related work that uses these algorithms for baseline comparison. We further show the results we get, using the same hyperparameter configuration, but using two different codebase implementations (note that these implementations are often used as baseline codebase to develop algorithms). We highlight that, depending on the codebase used, experimental results can vary significantly. 

Table 5: Comparison with Related Reported Results with Hopper Environment 

Environment Metric rllab QProp IPG TRPO Our Results (rllab) Our Results (Baselines) TRPO on Hopper Environment Number of Iterations 500 500 500 500 500 500 Average Return 1183.3 - - - 2021.34 2965.3 Max Average Return - 2486 3668.8 3229.1 3034.4 Table 6: Comparison with Related Reported Results with HalfCheetah Environment 

Environment Metric rllab QProp IPG TRPO Our Results (rllab) Our Results (Baselines) TRPO on HalfCheetah Environment Number of Iterations 500 500 500 500 500 500 Average Return 1914.0 - - 3576.08 1045.6 Max Average Return - 4734 2889 4855 5197 1045.6 

Work Number of Trials (Mnih et al. 2016) top-5 (Schulman et al. 2017) 3-9 (Duan et al. 2016) 5 (5) (Gu et al. 2017) 3(Lillicrap et al. 2015b) 5(Schulman et al. 2015a) 5(Wu et al. 2017) top-2, top-3 Table 7: Number of trials reported during evaluation in various works. 

Reported Evaluation Metrics in Related Work 

In table 8 we show the evaluation metrics, and reported results in further details across related work. 

Table 8: Reported Evaluation Metrics of baseline algorithms in related literature 

Related Work (Algorithm) Environments Timesteps or Episodes or Iterations Evaluation Metrics Average Return Max Return Std Error 

PPO HalfCheetah Hopper 1M ∼1800 

∼2200 - -ACKTR HalfCheetah Hopper 1M ∼2400 

∼3500 - -Q-Prop (DDPG) HalfCheetah Hopper 6k (eps) ∼6000 -7490 2604 --Q-Prop (TRPO) HalfCheetah Hopper 5k (timesteps) ∼4000 -4734 2486 --IPG (TRPO) HalfCheetah Hopper 10k (eps) ∼3000 - 2889 --Param Noise (DDPG) HalfCheetah Hopper 1M ∼1800 

∼500 ----Param Noise (TRPO) HalfCheetah Hopper 1M ∼3900 

∼2400 ----Benchmarking (DDPG) HalfCheetah Hopper 500 iters (25k eps) 

∼2148 

∼267 --702 43 Benchmarking (TRPO) HalfCheetah Hopper 500 iters (925k eps) 

∼1914 

∼1183 --150 120 

## Experimental Setup 

In this section, we show detailed analysis of our experimental results, using same hyperparameter configurations used in related work. Experimental results are included for the OpenAI Gym (Brockman et al . 2016) Hopper-v1 and HalfCheetah-v1 environments, using the policy gradient algorithms including DDPG, TRPO, PPO and ACKTR. Our experiments are done using the available codebase from OpenAI rllab (Duan et al . 2016) and OpenAI Baselines. Each of our experiments are performed over 5 experimental trials with different random seeds, and results averaged over all trials. Unless explicitly specified as otherwise (such as in hyperparameter modifications where we alter a hyperparameter under investigation), hyperparameters were as follows. All results (including graphs) show mean and standard error across random seeds. 

• DDPG – Policy Network: (64, relu, 64, relu, tanh); Q Network (64, relu, 64, relu, linear) 

– Normalized observations with running mean filter 

– Actor LR: 1e − 4; Critic LR: 1e − 3

– Reward Scale: 1.0 

– Noise type: O-U 0.2 

– Soft target update τ = .01 

– γ = 0 .995 

– batch size = 128 

– Critic L2 reg 1e − 2

• PPO 

– Policy Network: (64, tanh, 64, tanh, Linear) + Standard Deviation variable; Value Network (64, tanh, 64, tanh, linear) 

– Normalized observations with running mean filter 

– Timesteps per batch 2048 

– clip param = 0.2 

– entropy coeff = 0.0 

– Optimizer epochs per iteration = 10 

– Optimizer step size 3e − 4

– Optimizer batch size 64 

– Discount γ = 0 .995 , GAE λ = 0 .97 

– learning rate schedule is constant 

• TRPO 

– Policy Network: (64, tanh, 64, tanh, Linear) + Standard Deviation variable; Value Network (64, tanh, 64, tanh, linear) 

– Normalized observations with running mean filter 

– Timesteps per batch 5000 

– max KL=0.01 

– Conjugate gradient iterations = 20 

– CG damping = 0.1 

– VF Iterations = 5 

– VF Batch Size = 64 

– VF Step Size = 1e − 3

– entropy coeff = 0.0 

– Discount γ = 0 .995 , GAE λ = 0 .97 

• ACKTR 

– Policy Network: (64, tanh, 64, tanh, Linear) + Standard Deviation variable; Value Network (64, elu, 64, elu, linear) 

– Normalized observations with running mean filter 

– Timesteps per batch 2500 

– desired KL = .002 

– Discount γ = 0 .995 , GAE λ = 0 .97 

Modifications to Baseline Implementations 

To ensure fairness of comparison, we make several modifications to the existing implementations. First, we change evaluation in DDPG (Plappert et al . 2017) such that during evaluation at the end of an epoch, 10 full trajectories are evaluated. In the current implementation, only a partial trajectory is evaluated immediately after training such that a full trajectory will be evaluated across several different policies, this corresponds more closely to the online view of evaluation, while we take a policy optimization view when evaluating algorithms. 

Hyperparameters : Network Structures and Activation Functions 

Below, we examine the significance of the network configurations used for the non-linear function approximators in policy gradient methods. Several related work have used different sets of network configurations (network sizes and activation functions). We use the reported network configurations from other works, and demonstrate the significance of careful fine tuning that is required. We demonstrate results using the network activation functions, ReLU, TanH and Leaky ReLU, where most papers use ReLU and TanH as activation functions without detailed reporting of the effect of these activation functions. We analyse the signifcance of using different activations in the policy and action value networks. Previously, we included a detailed table showing average reward with standard error obtained for each of the hyperparameter configurations. In the results below, we show detailed results of how each of these policy gradient algorithms are affected by the choice of the network configuration. Proximal Policy Optimization (PPO) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 0500 1000 1500 2000 2500 

> Average Return

HalfCheetah-v1 (PPO, Value Network Activation) 

> tanh relu leaky relu

Figure 7: PPO Policy and Value Network activation 

Experiment results in Figure 7, 8, and 9 in this section show the effect of the policy network structures and activation functions in the Proximal Policy Optimization (PPO) algorithm. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−2000 

−1000 

0

1000 

2000 

> Average Return

HalfCheetah-v1 (PPO, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

Figure 8: PPO Policy Network structure Figure 9: PPO Value Network structure 

Actor Critic using Kronecker-Factored Trust Region (ACKTR) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

HalfCheetah-v1 (ACKTR, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

Hopper-v1 (ACKTR, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

Figure 10: ACKTR Policy Network structure 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

HalfCheetah-v1 (ACKTR, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

Hopper-v1 (ACKTR, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

Figure 11: ACKTR Value Network structure 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

HalfCheetah-v1 (ACKTR, Policy Network Activation) 

> tanh
> relu
> elu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

Hopper-v1 (ACKTR, Policy Network Activation) 

> tanh
> relu
> elu

Figure 12: ACKTR Policy Network Activation 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

1000 

2000 

3000 

4000 

> Average Return

HalfCheetah-v1 (ACKTR, Value Network Activation) 

> tanh
> relu
> elu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

2000 

2500 

3000 

> Average Return

Hopper-v1 (ACKTR, Value Network Activation) 

> tanh
> relu
> elu

Figure 13: ACKTR Value Network Activation 

We then similarly, show the significance of these hyperparameters in the ACKTR algorithm. Our results show that the value network structure can have a significant effect on the performance of ACKTR algorithm. 

Trust Region Policy Optimization (TRPO) 

Figure 14: TRPO Policy Network structure Figure 15: TRPO Value Network structure 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> −750
> −500
> −250
> 0
> 250
> 500
> 750
> 1000
> Average Return

HalfCheetah-v1 (TRPO, Policy Network Activation) 

> tanh
> relu
> leaky relu

Figure 16: TRPO Policy and Value Network activation 

Figure 17: TRPO Policy and Value Network activation 

In Figures 14, 15, 16, and 17 we show the effects of network structure on the OpenAI baselines implementation of TRPO. In this case, only the policy architecture seems to have a large effect on the performance of the algorithm’s ability to learn. Deep Deterministic Policy Gradient (DDPG) 

Figure 18: Policy or Actor Network Architecture experiments for DDPG on HalfCheetah and Hopper Environment 

We further analyze the actor and critic network configurations for use in DDPG. As in default configurations, we first use the ReLU activation function for policy networks, and examine the effect of different activations and network sizes for the critic networks. Similarly, keeping critic network configurations under default setting, we also examine the effect of actor network activation functions and network sizes. Figure 19: Significance of Value Function or Critic Network Activations for DDPG on HalfCheetah and Hopper Environment 

## Reward Scaling Parameter in DDPG 

Figure 20: DDPG reward rescaling on Hopper-v1, with and without layer norm. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return

HalfCheetah-v1 (DDPG, Reward Scale, Layer Norm)         

> rs=1e-4
> rs=1e-3
> rs=1e-2
> rs=1e-1
> rs=1
> rs=10
> rs=100 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 1000
> 2000
> 3000
> Average Return

HalfCheetah-v1 (DDPG, Reward Scale, No Layer Norm) 

> rs=1e-4
> rs=1e-3
> rs=1e-2
> rs=1e-1
> rs=1
> rs=10
> rs=100

Figure 21: DDPG reward rescaling on HalfCheetah-v1, with and without layer norm. 

Several related work (Gu et al . 2016; 2017; Duan et al . 2016) have often reported that for DDPG the reward scaling parameter often needs to be fine-tuned for stabilizing the performance of DDPG. It can make a significant impact in performance of DDPG based on the choice of environment. We examine several reward scaling parameters and demonstrate the effect this parameter can have on the stability and performance of DDPG, based on the HalfCheetah and Hopper environments. Our experiment results, as demonstrated in Figure 21 and 20, show that the reward scaling parameter indeed can have a significant impact on performance. Our results show that, very small or negligible reward scaling parameter can significantly detriment the performance of DDPG across all environments. Furthermore, a scaling parameter of 10 or 1 often performs good. Based on our analysis, we suggest that every time DDPG is reported as a baseline algorithm for comparison, the reward scaling parameter should be fine-tuned, specific to the algorithm. Batch Size in TRPO 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> 3500
> Average Return

Hopper-v1 (TRPO, original, Batch Size)         

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> −500
> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> Average Return

HalfCheetah-v1 (TRPO, original, Batch Size) 

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768

Figure 22: TRPO (Schulman et al. 2015a) original code batch size experiments. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> Average Return

Hopper-v1 (TRPO, baselines, Batch Size)         

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> −600
> −400
> −200
> 0
> 200
> Average Return

HalfCheetah-v1 (TRPO, baselines, Batch Size)         

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> Average Return

Walker2d-v1 (TRPO, baselines, Batch Size)         

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> −135
> −130
> −125
> −120
> −115
> −110
> Average Return

Reacher-v1 (TRPO, baselines, Batch Size) 

> 1024
> 2048
> 4096
> 8192
> 16384
> 32768

Figure 23: TRPO (Schulman et al. 2017) baselines code batch size experiments. 

We run batch size experiments using the original TRPO code (Schulman et al . 2015a) and the OpenAI baselines code (Schulman et al . 2017). These results can be found in Experiment results in Figure 22 and Figure 23, show that for both HalfCheetah-v1 and Hopper-v1 environments, a batch size of 1024 for TRPO performs best, while perform degrades consecutively as the batch size is increased. 

## Random Seeds 

To determine much random seeds can affect results, we run 10 trials total on two environments using the default previously described settings usign the (Gu et al . 2016) implementation of DDPG and the (Duan et al . 2016) version of TRPO. We divide our trials random into 2 partitions and plot them in Figures 24 and Fig 25. As can be seen, statistically different distributions can be attained just from the random seeds with the same exact hyperparameters. As we will discuss later, bootstrapping off of the sample can give an idea for how drastic this effect will be, though too small a bootstrap will still not give concrete enough results. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return

HalfCheetah-v1 (TRPO, Different Random Seeds)         

> Random Average (5 runs)
> Random Average (5 runs) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> 3500
> Average Return

Hopper-v1 (TRPO, Different Random Seeds) 

> Random Average (5 runs)
> Random Average (5 runs)

Figure 24: Two different TRPO experiment runs, with same hyperparameter configurations, averaged over two splits of 5 different random seeds. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 1000
> 2000
> 3000
> 4000
> Average Return

HalfCheetah-v1 (DDPG, Different Random Seeds)         

> Random Average (5 runs)
> Random Average (5 runs) 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 250
> 500
> 750
> 1000
> 1250
> 1500
> 1750
> Average Return

Hopper-v1 (DDPG, Different Random Seeds) 

> Random Average (5 runs)
> Random Average (5 runs)

Figure 25: Two different DDPG experiment runs, with same hyperparameter configurations, averaged over two splits of 5 different random seeds. 

## Choice of Benchmark Continuous Control Environment 

We previously demonstrated that the performance of policy gradient algorithms can be highly biased based on the choice of the environment. In this section, we include further results examining the impact the choice of environment can have. We show that no single algorithm can perform consistenly better in all environments. This is often unlike the results we see with DQN networks in Atari domains, where results can often be demonstrated across a wide range of Atari games. Our results, for example, shows that while TRPO can perform significantly better than other algorithms on the Swimmer environment, it may perform quite poorly n the HalfCheetah environment, and marginally better on the Hopper environment compared to PPO. We demonstrate our results using the OpenAI MuJoCo Gym environments including Hopper, HalfCheetah, Swimmer and Walker environments. It is notable to see the varying performance these algorithms can have even in this small set of environment domains. The choice of reporting algorithm performance results can therefore often be biased based on the algorithm designer’s experience with these environments. Figure 26: Comparing Policy Gradients across various environments 

## Codebases 

We include a detailed analysis of performance comparison, with different network structures and activations, based on the choice of the algorithm implementation codebase. 

Figure 27: TRPO Policy and Value Network structure Figure 28: TRPO Policy and Value Network activations. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

−250 

0

250 

500 

750 

1000 

1250 

> Average Return

HalfCheetah-v1 (TRPO, rllab, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

1200 

1400 

> Average Return

Hopper-v1 (TRPO, rllab, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

−250 

0

250 

500 

750 

1000 

1250 

1500 

> Average Return

HalfCheetah-v1 (TRPO, rllab, Policy Network Activation) 

> tanh
> relu
> leaky relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

1200 

1400 

> Average Return

Hopper-v1 (TRPO, rllab, Policy Network Activation) 

> tanh
> relu
> leaky relu

Figure 29: TRPO rllab Policy Structure and Activation 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

3500 

> Average Return

HalfCheetah-v1 (DDPG, rllab++, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

100 

200 

300 

400 

500 

600 

700 

800 

> Average Return

Hopper-v1 (DDPG, rllab++, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

1200 

> Average Return

Hopper-v1 (DDPG, rllab++, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

> Average Return

HalfCheetah-v1 (DDPG, rllab++, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

Figure 30: DDPG rllab++ Policy and Value Network structure 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

3500 

> Average Return

HalfCheetah-v1 (DDPG, rllab++, Policy Network Activation) 

> tanh
> relu
> leaky relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

> Average Return

Hopper-v1 (DDPG, rllab++, Policy Network Activation) 

> tanh
> relu
> leaky

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

> Average Return

HalfCheetah-v1 (DDPG, rllab++, Value Network Activation) 

> tanh
> relu
> leaky relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

> Average Return

Hopper-v1 (DDPG, rllab++, Value Network Activation) 

> tanh
> relu
> leaky relu

Figure 31: DDPG rllab++ Policy and Value Network activations. 

Similarly, Figures 32 and 33 show the same network experiments for DDPG with the Theano implementation of rllab code (Duan et al .2016). 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

2500 

3000 

3500 

> Average Return

HalfCheetah-v1 (DDPG, rllab, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

100 

200 

300 

400 

500 

600 

700 

800 

> Average Return

Hopper-v1 (DDPG, rllab, Policy Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

250 

500 

750 

1000 

1250 

1500 

1750 

2000 

> Average Return

Hopper-v1 (DDPG, rllab, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

−500 

0

500 

1000 

1500 

2000 

> Average Return

HalfCheetah-v1 (DDPG, rllab, Value Network Structure) 

> (64,64)
> (100,50,25)
> (400,300)

Figure 32: DDPG rllab Policy and Value Network structure 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

2000 

> Average Return

HalfCheetah-v1 (DDPG, rllab, Policy Network Activation) 

> tanh
> relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

> Average Return

Hopper-v1 (DDPG, rllab, Policy Network Activation) 

> tanh
> relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

500 

1000 

1500 

> Average Return

HalfCheetah-v1 (DDPG, rllab, Value Network Activation) 

> tanh
> relu

0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

0

200 

400 

600 

800 

1000 

> Average Return

Hopper-v1 (DDPG, rllab, Value Network Activation) 

> tanh
> relu

Figure 33: DDPG rllab Policy and Value Network activations. 

Often in related literature, there is different baseline codebase people use for implementation of algorithms. One such example is for the TRPO algorithm. It is a commonly used policy gradient method for continuous control tasks, and there exists several implementations from OpenAI Baselines (Plappert et al . 2017), OpenAI rllab (Duan et al . 2016) and the original TRPO codebase (Schulman et al . 2015a). In this section, we perform an analysis of the impact the choice of algorithm codebase can have on the performance. Figures 27 and 28 summarizes our results with TRPO policy network and value networks, using the original TRPO codebase from (Schulman et al . 2015a). Figure 29 shows the results using the rllab implementation of TRPO using the same hyperparameters as our default experiments aforementioned. Note, we use a linear function approximator rather than a neural network due to the fact that the Tensorflow implementation of OpenAI rllab doesn’t provide anything else. We note that this is commonly used in other works (Duan et al . 2016; Stadie, Abbeel, and Sutskever 2017), but may cause differences in performance. Furthermore, we leave out our value function network experiments due to this. 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> 0
> 1000
> 2000
> 3000
> 4000
> 5000
> Average Return

HalfCheetah-v1 (DDPG, Codebase Comparison)         

> Duan 2016
> Gu 2016
> Plapper 2017 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 250
> 500
> 750
> 1000
> 1250
> 1500
> 1750
> Average Return

Hopper-v1 (DDPG, Codebase Comparison) 

> Duan 2016
> Gu 2016
> Plapper 2017

Figure 34: DDPG codebase comparison using our default set of hyperparameters (as used in other experiments). 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 

Timesteps ×10 6

> −500
> 0
> 500
> 1000
> 1500
> 2000
> Average Return

HalfCheetah-v1 (TRPO, Codebase Comparison)         

> Schulman 2015
> Schulman 2017
> Duan 2016 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00

Timesteps ×10 6

> 0
> 500
> 1000
> 1500
> 2000
> 2500
> 3000
> Average Return

Hopper-v1 (TRPO, Codebase Comparison) 

> Schulman 2015
> Schulman 2017
> Duan 2016

Figure 35: TRPO codebase comparison using our default set of hyperparameters (as used in other experiments). 

Figure 35 shows a comparison of the TRPO implementations using the default hyperparamters as specified earlier in the supplemental. Note, the exception is that we use a larger batch size for rllab and original TRPO code of 20k samples per batch, as optimized in a second set of experiments. Figure 30 and 31 show the same network experiments for DDPG with the rllab++ code (Gu et al . 2016). We can then compare the performance of the algorithm across 3 codebases (keeping all hyperparameters constant at the defaults), this can be seen in Figure 34. 

## Significance 

Our full results from significance testing with difference metrics can be found in Table 9 and Table 10. Our bootstrap mean and confidence intervals can be found in Table 13. Bootstrap power analysis can be found in Table 14. To performance significance testing, we use our 5 sample trials to generate a bootstrap with 10k bootstraps. From this confidence intervals can be obtained. For the t-test and KS-test, the average returns from the 5 trials are sorted and compared using the normal 2-sample versions of these tests. Scipy ( https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp. html , https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html ) and Facebook Boostrapped ( https://github.com/facebookincubator/bootstrapped ) are used for the KS test, t-test, and bootstrap analysis. For power analysis, we attempt to determine if a sample is enough to game the significance of a 25% lift. This is commonly used in A/B testing (Tuff´ ery 2011). - DDPG ACKTR TRPO PPO DDPG -

t = 1 .85 , p = 0 .102 

KS = 0 .60 , p = 0 .209 

61.91 % (-32.27 %, 122.99 %) 

t = 4 .59 , p = 0 .002 

KS = 1 .00 , p = 0 .004 

301.48 % (150.50 %, 431.67 %) 

t = 2 .67 , p = 0 .029 

KS = 0 .80 , p = 0 .036 

106.91 % (-37.62 %, 185.26 %) ACKTR 

t = −1.85 , p = 0 .102 

KS = 0 .60 , p = 0 .209 

-38.24 % (-75.42 %, -15.19 %) -

t = 2 .78 , p = 0 .024 

KS = 0 .80 , p = 0 .036 

147.96 % (30.84 %, 234.60 %) 

t = 0 .80 , p = 0 .448 

KS = 0 .60 , p = 0 .209 

27.79 % (-67.77 %, 79.56 %) TRPO 

t = −4.59 , p = 0 .002 

KS = 1 .00 , p = 0 .004 

-75.09 % (-86.44 %, -68.36 %) 

t = −2.78 , p = 0 .024 

KS = 0 .80 , p = 0 .036 

-59.67 % (-81.70 %, -46.84 %) -

t = −2.12 , p = 0 .067 

KS = 0 .80 , p = 0 .036 

-48.46 % (-81.23 %, -32.05 %) PPO 

t = −2.67 , p = 0 .029 

KS = 0 .80 , p = 0 .036 

-51.67 % (-80.69 %, -31.94 %) 

t = −0.80 , p = 0 .448 

KS = 0 .60 , p = 0 .209 

-21.75 % (-75.99 %, 11.68 %) 

t = 2 .12 , p = 0 .067 

KS = 0 .80 , p = 0 .036 

94.04 % (2.73 %, 169.06 %) -

Table 9: HalfCheetah Significance values and metrics for different algorithms. Rows in cells are: sorted 2-sample t-test, Kolmogorov-Smirnov test, bootstrap A/B comparison % difference with 95% confidence bounds. 

- DDPG ACKTR TRPO PPO DDPG -

t = −1.41 , p = 0 .196 

KS = 0 .60 , p = 0 .209 

-35.92 % (-85.62 %, -5.38 %) 

t = −2.58 , p = 0 .033 

KS = 0 .80 , p = 0 .036 

-44.96 % (-78.82 %, -20.29 %) 

t = −2.09 , p = 0 .070 

KS = 0 .80 , p = 0 .036 

-39.90 % (-77.12 %, -12.95 %) ACKTR 

t = 1 .41 , p = 0 .196 

KS = 0 .60 , p = 0 .209 

56.05 % (-87.98 %, 123.15 %) -

t = −1.05 , p = 0 .326 

KS = 0 .60 , p = 0 .209 

-14.11 % (-37.17 %, 9.11 %) 

t = −0.42 , p = 0 .686 

KS = 0 .40 , p = 0 .697 

-6.22 % (-31.58 %, 18.98 %) TRPO 

t = 2 .58 , p = 0 .033 

KS = 0 .80 , p = 0 .036 

81.68 % (-67.76 %, 151.64 %) 

t = 1 .05 , p = 0 .326 

KS = 0 .60 , p = 0 .209 

16.43 % (-27.92 %, 41.17 %) -

t = 2 .57 , p = 0 .033 

KS = 0 .60 , p = 0 .209 

9.19 % (2.37 %, 15.58 %) PPO 

t = 2 .09 , p = 0 .070 

KS = 0 .80 , p = 0 .036 

66.39 % (-67.80 %, 130.16 %) 

t = 0 .42 , p = 0 .686 

KS = 0 .40 , p = 0 .697 

6.63 % (-33.54 %, 29.59 %) 

t = −2.57 , p = 0 .033 

KS = 0 .60 , p = 0 .209 

-8.42 % (-14.08 %, -2.97 %) -

Table 10: Hopper Significance values and metrics for different algorithms. Rows in cells are: sorted 2-sample t-test, Kolmogorov-Smirnov test, bootstrap A/B comparison % difference with 95% confidence bounds. 

- DDPG ACKTR TRPO PPO DDPG -

t = −1.03 , p = 0 .334 

KS = 0 .40 , p = 0 .697 

-30.78 % (-91.35 %, 1.06 %) 

t = −4.04 , p = 0 .004 

KS = 1 .00 , p = 0 .004 

-48.52 % (-70.33 %, -28.62 %) 

t = −3.07 , p = 0 .015 

KS = 0 .80 , p = 0 .036 

-45.95 % (-70.85 %, -24.65 %) ACKTR 

t = 1 .03 , p = 0 .334 

KS = 0 .40 , p = 0 .697 

44.47 % (-80.62 %, 111.72 %) -

t = −1.35 , p = 0 .214 

KS = 0 .60 , p = 0 .209 

-25.63 % (-61.28 %, 5.54 %) 

t = −1.02 , p = 0 .338 

KS = 0 .60 , p = 0 .209 

-21.91 % (-61.53 %, 11.02 %) TRPO 

t = 4 .04 , p = 0 .004 

KS = 1 .00 , p = 0 .004 

94.24 % (-22.59 %, 152.61 %) 

t = 1 .35 , p = 0 .214 

KS = 0 .60 , p = 0 .209 

34.46 % (-60.47 %, 77.32 %) -PPO 

t = 3 .07 , p = 0 .015 

KS = 0 .80 , p = 0 .036 

85.01 % (-31.02 %, 144.35 %) 

t = 1 .02 , p = 0 .338 

KS = 0 .60 , p = 0 .209 

28.07 % (-65.67 %, 71.71 %) 

t = −0.57 , p = 0 .582 

KS = 0 .40 , p = 0 .697 

-4.75 % (-19.06 %, 10.02 %) -

Table 11: Walker2d Significance values and metrics for different algorithms. Rows in cells are: sorted 2-sample t-test, Kolmogorov-Smirnov test, bootstrap A/B comparison % difference with 95% confidence bounds. 

- DDPG ACKTR TRPO PPO DDPG -

t = −2.18 , p = 0 .061 

KS = 0 .80 , p = 0 .036 

-36.44 % (-61.04 %, -6.94 %) 

t = −4.06 , p = 0 .004 

KS = 1 .00 , p = 0 .004 

-85.13 % (-97.17 %, -77.95 %) 

t = −8.33 , p = 0 .000 

KS = 1 .00 , p = 0 .004 

-70.41 % (-80.86 %, -56.52 %) ACKTR 

t = 2 .18 , p = 0 .061 

KS = 0 .80 , p = 0 .036 

57.34 % (-80.96 %, 101.11 %) -

t = −3.69 , p = 0 .006 

KS = 1 .00 , p = 0 .004 

-76.61 % (-90.68 %, -70.06 %) 

t = −8.85 , p = 0 .000 

KS = 1 .00 , p = 0 .004 

-53.45 % (-62.22 %, -47.30 %) TRPO 

t = 4 .06 , p = 0 .004 

KS = 1 .00 , p = 0 .004 

572.61 % (-73.29 %, 869.24 %) 

t = 3 .69 , p = 0 .006 

KS = 1 .00 , p = 0 .004 

327.48 % (165.47 %, 488.66 %) -

t = 2 .39 , p = 0 .044 

KS = 0 .60 , p = 0 .209 

99.01 % (28.44 %, 171.85 %) PPO 

t = 8 .33 , p = 0 .000 

KS = 1 .00 , p = 0 .004 

237.97 % (-59.74 %, 326.85 %) 

t = 8 .85 , p = 0 .000 

KS = 1 .00 , p = 0 .004 

114.80 % (81.85 %, 147.33 %) 

t = −2.39 , p = 0 .044 

KS = 0 .60 , p = 0 .209 

-49.75 % (-78.58 %, -36.43 %) -

Table 12: Swimmer Significance values and metrics for different algorithms. Rows in cells are: sorted 2-sample t-test, Kolmogorov-Smirnov test, bootstrap A/B comparison % difference with 95% confidence bounds. Environment DDPG ACKTR TRPO PPO HalfCheetah-v1 5037.26 (3664.11, 6574.01) 3888.85 (2288.13, 5131.96) 1254.55 (999.52, 1464.86) 3043.1 (1920.4, 4165.86) Hopper-v1 1632.13 (607.98, 2370.21) 2546.89 (1875.79, 3217.98) 2965.33 (2854.66, 3076.00) 2715.72 (2589.06, 2847.93) Walker2d-v1 1582.04 (901.66, 2174.66) 2285.49 (1246.00, 3235.96) 3072.97 (2957.94, 3183.10) 2926.92 (2514.83, 3361.43) Swimmer-v1 31.92 (21.68, 46.23) 50.22 (42.47, 55.37) 214.69 (141.52, 287.92) 107.88 (101.13, 118.56) 

Table 13: Envs bootstrap mean and 95% confidence bounds Environment DDPG ACKTR TRPO PPO HalfCheetah-v1 100.00 % 0.00 % 0.00 % 79.03 % 11.53 % 9.43 % 79.47 % 20.53 % 0.00 % 61.07 % 10.50 % 28.43 % Hopper-v1 60.90 % 10.00 % 29.10 % 79.60 % 11.00 % 9.40 % 0.00 % 100.00 % 0.00 % 0.00 % 100.00 % 0.00 % Walker2d-v1 89.50 % 0.00 % 10.50 % 60.33 % 9.73 % 29.93 % 0.00 % 100.00 % 0.00 % 59.80 % 31.27 % 8.93 % Swimmer-v1 89.97 % 0.00 % 10.03 % 59.90 % 40.10 % 0.00 % 89.47 % 0.00 % 10.53 % 40.27 % 59.73 % 0.00 % Table 14: Power Analysis for predicted significance of 25% lift. Rows in cells are: % insignificant simulations,% positive significant, % negative significant.
