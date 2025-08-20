# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Hashable

import networkx as nx
import pandas as pd

import cyclebane as cb


def idx(
    name: str, *index: Hashable, offset=None, dims: tuple[str, ...] = ('dim_0', 'dim_1')
) -> cb.graph.NodeName:
    """Helper to create a NodeName with a tuple of indices."""
    return cb.graph.NodeName(
        name,
        cb.graph.IndexValues(dims[offset : len(index) + (offset or 0)], tuple(index)),
    )


def test_basic_map_groupby_reduce_gives_correct_graph_structure() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')
    df = pd.DataFrame({'a': [11, 22, 33], 'b': ['a', 'a', 'b']})

    graph = cb.Graph(g)
    mapped = graph.map(df)
    grouped = mapped.groupby('b').reduce('c', name='d')

    result = grouped.to_networkx()

    # Nodes before grouping
    assert result.nodes[idx('a', 0)] == {'value': 11}
    assert result.nodes[idx('b', 0)] == {'value': 'a'}
    assert result.nodes[idx('c', 0)] == {}
    # Nodes after grouping
    assert result.nodes[idx('d', 'a', dims=('b',))] == {}

    # Edges to grouped node
    assert result.has_edge(idx('c', 0), idx('d', 'a', dims=('b',)))
    assert result.has_edge(idx('c', 1), idx('d', 'a', dims=('b',)))
    assert result.has_edge(idx('c', 2), idx('d', 'b', dims=('b',)))
    # No cross-group edges
    assert not result.has_edge(idx('c', 0), idx('d', 'b', dims=('b',)))
    assert not result.has_edge(idx('c', 1), idx('d', 'b', dims=('b',)))
    assert not result.has_edge(idx('c', 2), idx('d', 'a', dims=('b',)))
