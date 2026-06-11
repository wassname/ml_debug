# How to avoid machine learning pitfalls: checklist

Appendix to the [ML Debugging skill](../SKILL.md).

This is the full do/don't list from Michael A. Lones, ["How to avoid machine learning pitfalls: a guide for academic researchers"](https://arxiv.org/abs/2108.02497) (v5, updated annually). Read the paper for the reasoning and examples behind each item; the local evidence excerpt is [here](../docs/evidence/lones_2021_ml_pitfalls.md).

> Mistakes in machine learning practice are commonplace, and can result in a loss of confidence in the findings and products of machine learning.

## Before you start to build models

> 2.1 Do think about how and where you will use data  
> 2.2 Do take the time to understand your data  
> 2.3 Don't look at all your data  
> 2.4 Do clean your data  
> 2.5 Do make sure you have enough data  
> 2.6 Do talk to domain experts  
> 2.7 Do survey the literature  
> 2.8 Do think about how your model will be deployed

## How to reliably build models

> 3.1 Don't allow test data to leak into the training process  
> 3.2 Do try out a range of different models  
> 3.3 Don't use inappropriate models  
> 3.4 Do keep up with progress in deep learning (and its pitfalls)  
> 3.5 Don't assume deep learning will be the best approach  
> 3.6 Do be careful where and how you do feature selection  
> 3.7 Do optimise your model's hyperparameters  
> 3.8 Do avoid learning spurious correlations

## How to robustly evaluate models

> 4.1 Do use an appropriate test set  
> 4.2 Don't do data augmentation before splitting your data  
> 4.3 Do avoid sequential overfitting  
> 4.4 Do evaluate a model multiple times  
> 4.5 Do save some data to evaluate your final model instance  
> 4.6 Do choose metrics carefully  
> 4.7 Do consider model fairness  
> 4.8 Don't ignore temporal dependencies in time series data

## How to compare models fairly

> 5.1 Don't assume a bigger number means a better model  
> 5.2 Do use meaningful baselines  
> 5.3 Do use statistical tests when comparing models  
> 5.4 Do correct for multiple comparisons  
> 5.5 Don't always believe results from community benchmarks  
> 5.6 Do combine models (carefully)

## How to report your results

> 6.1 Do be transparent  
> 6.2 Do report performance in multiple ways  
> 6.3 Don't generalise beyond the data  
> 6.4 Do be careful when reporting statistical significance  
> 6.5 Do look at your models  
> 6.6 Do use a machine learning checklist

Two especially common leak routes:

> The best thing you can do to prevent these issues is to partition off a subset of your data right at the start of your project, and only use this independent test set once to measure the generality of a single model at the end.

> Most notably, time series data are subject to a particular kind of data leakage known as look ahead bias.
