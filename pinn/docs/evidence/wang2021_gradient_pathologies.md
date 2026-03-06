# Wang et al. 2021 -- Understanding and Mitigating Gradient Flow Pathologies in Physics-informed Neural Networks
Source: https://arxiv.org/abs/2001.04536

## Paper Metadata

- **Authors:** Sifan Wang, Yujun Teng, Paris Perdikaris
- **Submitted:** 13 Jan 2020
- **arXiv ID:** 2001.04536 [cs.LG]
- **DOI:** https://doi.org/10.48550/arXiv.2001.04536
- **Length:** 28 pages, 18 figures
- **Subjects:** Machine Learning (cs.LG); Numerical Analysis (math.NA); Machine Learning (stat.ML)
- **Code & Data:** https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs

## Abstract

The widespread use of neural networks across different scientific domains often involves constraining them to satisfy certain symmetries, conservation laws, or other domain knowledge. Such constraints are often imposed as soft penalties during model training and effectively act as domain-specific regularizers of the empirical risk loss. Physics-informed neural networks is an example of this philosophy in which the outputs of deep neural networks are constrained to approximately satisfy a given set of partial differential equations.

In this work we review recent advances in scientific machine learning with a specific focus on the effectiveness of physics-informed neural networks in predicting outcomes of physical systems and discovering hidden physics from noisy data. We will also identify and analyze a fundamental mode of failure of such approaches that is related to numerical stiffness leading to unbalanced back-propagated gradients during model training.

To address this limitation we present a learning rate annealing algorithm that utilizes gradient statistics during model training to balance the interplay between different terms in composite loss functions. We also propose a novel neural network architecture that is more resilient to such gradient pathologies.

Taken together, our developments provide new insights into the training of constrained neural networks and consistently improve the predictive accuracy of physics-informed neural networks by a factor of 50-100x across a range of problems in computational physics.

## Key Contributions

1. **Problem Identification:** Identifies gradient flow pathologies in PINNs arising from numerical stiffness, causing unbalanced back-propagated gradients
2. **Learning Rate Annealing:** Proposes an algorithm using gradient statistics to balance different loss components during training
3. **Novel Architecture:** Introduces a new neural network architecture more resilient to gradient pathologies
4. **Empirical Results:** Demonstrates 50-100x improvement in predictive accuracy across computational physics problems

## Access

- PDF: https://arxiv.org/pdf/2001.04536
- TeX Source: https://arxiv.org/src/2001.04536
