# Wasserplan
`wasserplan` is a library for computing 1-Wasserstein distances between [GerryChain](https://github.com/mggg/gerrychain) partitions.

# Dependencies
`wasserplan` depends on GerryChain and [CVXPY](https://www.cvxpy.org/). We recommend installing these dependencies with Anaconda (see the [official GerryChain installation instructions](https://github.com/mggg/gerrychain#installation)).

## Using
We principally use this library to compute pairwise transport distances which we (approximately) project into 2D Euclidean space. A simple demonstration of this use case using Virginia congressional districts is available as a notebook (open in Colab).

## Performance
By default, `wasserplan` uses the open-source, lightweight [ECOS solver](https://github.com/embotech/ecos), which is included with CVXPY. However, this solver may exhibit performance issues for large graphs (thousands of nodes). Performance can be improved by using a commercial solver such as [MOSEK](https://www.mosek.com/).

## License
This library is available under the MIT License.
