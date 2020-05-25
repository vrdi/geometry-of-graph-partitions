# Geometry of Graph Partitions via Optimal Transport

This repository contains source code and data for the experiments in our paper ["Geometry of Graph Partitions via Optimal Transport"](https://arxiv.org/abs/1910.09618) (to appear in SISC). In the paper, we develop an optimal transport-based metric to measure the distance between graph partitions and apply the metric to problems in redistricting. This repository focuses exclusively on these redistricting applications.

## Wasserplan
`wasserplan` is a small Python library that uses [CVXPY](https://www.cvxpy.org/) to compute pairwise distances between [GerryChain](https://github.com/mggg/gerrychain) districting plans. It is [available on PyPI](https://pypi.org/project/wasserplan/).

## Experiments
In our paper, we examine several use cases for our metric using state geographies.

### Ensemble comparison and outlier analysis (Iowa)
In these experiments, we demonstrate the difference between the single-flip chain and the ReCom chain. The ReCom chain takes larger steps through the space of plans, and this is reflected in the relatively large distances between plans in the ReCom ensemble. We also show how geographical outliers (that is, plans near the edges of an embedded ensemble) tend to be political outliers.


### Ensemble comparison and outlier analysis (North Carolina)
In this experiment, we compare an ensemble generated from a ReCom chain to an ensemble generated from a flip-based chain created by Jonathan Mattingly's [Quantifying Redistricting](https://sites.duke.edu/quantifyinggerrymandering/) project at Duke. We demonstrate that outlying plans tend to be consistent for both chains. Notably, the enacted 2012 and 2016 plans are outliers with respect to both ensembles, and the plan proposed by a team of judges is not considered an outlier with respect to either ensemble.

### Simulated annealing (Arkansas)
In this experiment, we use `wasserplan` to visualize a chain of districting plans generated with simulated annealing. The chain is initially unconstrained by Metropolis weighting; later steps in the chain are weighted to encourage compactness. This experiment uses [Thomas Weighill's implementation of the landmark MDS algorithm](https://github.com/thomasweighill/landmarkMDS).


