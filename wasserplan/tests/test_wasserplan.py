import pytest
import numpy as np
from gerrychain import Partition
from gerrychain.grid import Grid
from wasserplan import Pair

@pytest.fixture
def horizontal_4x4():
    """Creates a 4x4 grid with 4 districts in horizontal stripes.

    Populations are assigned proportional to a nodes' horizontal position
    such that every district has equal population.

    3333
    2222
    1111
    0000
    """
    node_assignment = {}
    node_population = {}
    for x in range(4):
        for y in range(4):
            node_assignment[(x, y)] = y
            node_population[(x, y)] = x + 1
    grid = Grid(dimensions=(4, 4), assignment=node_assignment)
    for node, pop in node_population.items():
        grid.graph.nodes[node]['population'] = pop
    return grid


@pytest.fixture
def vertical_4x4():
    """Creates a 4x4 grid with 4 districts in vertical stripes.

    Populations are assigned proportional to a nodes' vertical position
    such that every district has equal population.

    0123
    0123
    0123
    0123
    """
    node_assignment = {}
    node_population = {}
    for x in range(4):
        for y in range(4):
            node_assignment[(x, y)] = x
            node_population[(x, y)] = y + 1
    grid = Grid(dimensions=(4, 4), assignment=node_assignment)
    for node, pop in node_population.items():
        grid.graph.nodes[node]['population'] = pop
    return grid


def test_node_embedding(horizontal_4x4, vertical_4x4):
    pair = Pair(horizontal_4x4, vertical_4x4)
    horizontal_embedding = pair._a_indicators
    vertical_embedding = pair._b_indicators

    # Embeddings sum to one (row-wise)
    assert np.array_equal(np.sum(horizontal_embedding, axis=1), np.ones(4))
    assert np.array_equal(np.sum(vertical_embedding, axis=1), np.ones(4))

    # Embeddings are pairwise unique
    assert not np.array_equal(horizontal_embedding, vertical_embedding)

    # Embeddings have one of two values: 1/<nodes per district> or 0
    assert np.array_equal(np.unique(horizontal_embedding), [0, 1/4])
    assert np.array_equal(np.unique(vertical_embedding), [0, 1/4])


def test_population_embedding(horizontal_4x4, vertical_4x4):
    pair = Pair(horizontal_4x4, vertical_4x4,
                indicator='population', pop_col='population')
    horizontal_embedding = pair._a_indicators
    vertical_embedding = pair._b_indicators

    # Embeddings sum to one (row-wise)
    assert np.array_equal(np.sum(horizontal_embedding, axis=1), np.ones(4))
    assert np.array_equal(np.sum(vertical_embedding, axis=1), np.ones(4))

    # Embeddings are pairwise unique
    assert not np.array_equal(horizontal_embedding, vertical_embedding)

    # Embedding values are proportional to grid coordinates or 0
    assert np.array_equal(np.unique(horizontal_embedding),
                          [idx / 10 for idx in range(5)])
    assert np.array_equal(np.unique(vertical_embedding),
                          [idx / 10 for idx in range(5)])
