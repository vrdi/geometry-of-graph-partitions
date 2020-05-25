Wasserplan
==========

``wasserplan`` is a library for computing 1-Wasserstein distances
between `GerryChain`_ partitions.

Dependencies
------------

``wasserplan`` depends on GerryChain and `CVXPY`_. We recommend
installing these dependencies with Anaconda (see the `official
GerryChain installation instructions`_).

Using
-----

We principally use this library to compute pairwise transport distances
which we (approximately) project into 2D Euclidean space. A simple
demonstration of this use case using Virginia congressional districts is
available `as a notebook`_ `(open in Colab)`_.

Performance
-----------

By default, ``wasserplan`` uses the open-source, lightweight `ECOS
solver`_, which is included with CVXPY. However, this solver may exhibit
performance issues for large graphs (thousands of nodes). Performance
can be improved by using a commercial solver such as `MOSEK`_.

License
-------

This library is available under the MIT License.

.. _GerryChain: https://github.com/mggg/gerrychain
.. _CVXPY: https://www.cvxpy.org/
.. _official GerryChain installation instructions: https://github.com/mggg/gerrychain#installation
.. _as a notebook: https://github.com/vrdi/geometry-of-graph-partitions/blob/master/wasserplan/VA%20ReCom%20demo.ipynb
.. _(open in Colab): https://colab.research.google.com/github/vrdi/geometry-of-graph-partitions/blob/master/wasserplan/VA%20ReCom%20demo.ipynb
.. _ECOS solver: https://github.com/embotech/ecos
.. _MOSEK: https://www.mosek.com/
