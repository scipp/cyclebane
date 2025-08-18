# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import networkx as nx
import pandas as pd

import cyclebane as cb


def idx(
    name: str, *index: int, offset=None, dims: tuple[str, ...] = ('dim_0', 'dim_1')
) -> cb.graph.NodeName:
    """Helper to create a NodeName with a tuple of indices."""
    return cb.graph.NodeName(
        name,
        cb.graph.IndexValues(dims[offset : len(index) + (offset or 0)], tuple(index)),
    )


def test_tmp() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')
    df = pd.DataFrame({'a': [11, 22, 33], 'b': ['a', 'a', 'b']})

    graph = cb.Graph(g)
    mapped = graph.map(df)
    grouped = mapped.groupby('b').reduce('c', name='d')
    print(grouped.graph.nodes)
    print(grouped.indices)
    result = grouped.to_networkx()
    for node in result.nodes:
        print(node, result.nodes[node])


def test_graphs_with_different_mapping_over_same_node_can_be_combined() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    result = mapped.to_networkx()

    assert result.nodes[idx('a', 0)] == {'value': 1}
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}
    assert result.nodes[idx('b', 0)] == {}
    assert result.nodes[idx('b', 1)] == {}
    assert result.nodes[idx('b', 2)] == {}
