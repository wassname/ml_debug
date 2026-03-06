# Rathore et al. 2024 -- Challenges in Training PINNs: A Loss Landscape Perspective

Source: https://arxiv.org/abs/2402.01868

## Citation Information

- Title: Challenges in Training PINNs: A Loss Landscape Perspective
- Authors: Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell
- arXiv ID: 2402.01868 (cs.LG)
- Submitted: 2 Feb 2024 (v1), last revised 3 Jun 2024 (v2)
- Venue: ICML 2024 Oral
- Pages: 33 pages (including appendices), 10 figures, 3 tables

## Abstract

This paper explores challenges in training Physics-Informed Neural Networks (PINNs), emphasizing the role of the loss landscape in the training process. We examine difficulties in minimizing the PINN loss function, particularly due to ill-conditioning caused by differential operators in the residual term. We compare gradient-based optimizers Adam, L-BFGS, and their combination Adam+L-BFGS, showing the superiority of Adam+L-BFGS, and introduce a novel second-order optimizer, NysNewton-CG (NNCG), which significantly improves PINN performance. Theoretically, our work elucidates the connection between ill-conditioned differential operators and ill-conditioning in the PINN loss and shows the benefits of combining first- and second-order optimization methods. Our work presents valuable insights and more powerful optimization strategies for training PINNs, which could improve the utility of PINNs for solving difficult partial differential equations.

## Key Claims and Contributions

1. **Problem Identification**: Ill-conditioning in PINN loss landscapes caused by differential operators in residual terms
2. **Optimizer Comparison**: Empirical evaluation of Adam, L-BFGS, and Adam+L-BFGS for PINN training
3. **Novel Method**: Introduction of NysNewton-CG (NNCG), a second-order optimizer with significant performance improvements
4. **Theoretical Connection**: Establishes link between ill-conditioned differential operators and ill-conditioning in PINN loss landscape
5. **Hybrid Optimization**: Demonstrates benefits of combining first-order and second-order optimization methods

## Metadata

- License: CC BY 4.0
- Subjects: Machine Learning (cs.LG); Optimization and Control (math.OC); Machine Learning (stat.ML)
- DOI: https://doi.org/10.48550/arXiv.2402.01868
- Available formats: PDF, HTML (experimental), TeX Source

## Access

- PDF: https://arxiv.org/pdf/2402.01868
- HTML: https://arxiv.org/html/2402.01868v2
- TeX Source: https://arxiv.org/src/2402.01868
