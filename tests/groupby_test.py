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


def test_group_twice_in_same_path() -> None:
    g1 = nx.DiGraph()
    g1.add_edge('a', 'c')
    g1.add_edge('param1', 'c')
    g1.add_edge('c', 'd')

    g2 = nx.DiGraph()
    g2.add_edge('e', 'f')
    g2.add_edge('param2', 'f')

    grouped = (
        cb.Graph(g1)
        .map(pd.DataFrame({'a': [11, 22, 33, 44], 'param1': ['x', 'x', 'y', 'z']}))
        .groupby('param1')
        .reduce('d', name='grouped-d')
    )
    mapped = cb.Graph(g2).map(
        pd.DataFrame(
            {'e': [1, 2, 3], 'param2': [0, 1, 1], 'param1': ['x', 'y', 'z']}
        ).set_index('param1')
    )

    mapped['e'] = grouped
    grouped_twice = mapped.groupby('param2').reduce('f', name='grouped-f')

    result = grouped_twice.to_networkx()

    # Nodes from second grouping (grouped-f)
    assert result.nodes[idx('grouped-f', 0, dims=('param2',))] == {}
    assert result.nodes[idx('grouped-f', 1, dims=('param2',))] == {}

    # Nodes from first grouping / mapping over param1
    assert result.nodes[idx('param2', 'x', dims=('param1',))] == {'value': 0}
    assert result.nodes[idx('param2', 'y', dims=('param1',))] == {'value': 1}
    assert result.nodes[idx('param2', 'z', dims=('param1',))] == {'value': 1}
    assert result.nodes[idx('f', 'x', dims=('param1',))] == {}
    assert result.nodes[idx('f', 'y', dims=('param1',))] == {}
    assert result.nodes[idx('f', 'z', dims=('param1',))] == {}
    # No value on 'e', was replaced by grouped-d links
    assert result.nodes[idx('e', 'x', dims=('param1',))] == {}
    assert result.nodes[idx('e', 'y', dims=('param1',))] == {}
    assert result.nodes[idx('e', 'z', dims=('param1',))] == {}
    assert idx('grouped-d', 'x', dims=('param1',)) not in result.nodes
    assert idx('grouped-d', 'y', dims=('param1',)) not in result.nodes
    assert idx('grouped-d', 'z', dims=('param1',)) not in result.nodes

    # Nodes from mapping over dim_0
    assert result.nodes[idx('a', 0)] == {'value': 11}
    assert result.nodes[idx('a', 1)] == {'value': 22}
    assert result.nodes[idx('a', 2)] == {'value': 33}
    assert result.nodes[idx('a', 3)] == {'value': 44}
    assert result.nodes[idx('param1', 0)] == {'value': 'x'}
    assert result.nodes[idx('param1', 1)] == {'value': 'x'}
    assert result.nodes[idx('param1', 2)] == {'value': 'y'}
    assert result.nodes[idx('param1', 3)] == {'value': 'z'}
    assert result.nodes[idx('c', 0)] == {}
    assert result.nodes[idx('c', 1)] == {}
    assert result.nodes[idx('c', 2)] == {}
    assert result.nodes[idx('c', 3)] == {}
    assert result.nodes[idx('d', 0)] == {}
    assert result.nodes[idx('d', 1)] == {}
    assert result.nodes[idx('d', 2)] == {}
    assert result.nodes[idx('d', 3)] == {}

    # Edges within dim_0 (original graph structure)
    assert result.has_edge(idx('a', 0), idx('c', 0))
    assert result.has_edge(idx('a', 1), idx('c', 1))
    assert result.has_edge(idx('a', 2), idx('c', 2))
    assert result.has_edge(idx('a', 3), idx('c', 3))
    assert result.has_edge(idx('param1', 0), idx('c', 0))
    assert result.has_edge(idx('param1', 1), idx('c', 1))
    assert result.has_edge(idx('param1', 2), idx('c', 2))
    assert result.has_edge(idx('param1', 3), idx('c', 3))
    assert result.has_edge(idx('c', 0), idx('d', 0))
    assert result.has_edge(idx('c', 1), idx('d', 1))
    assert result.has_edge(idx('c', 2), idx('d', 2))
    assert result.has_edge(idx('c', 3), idx('d', 3))

    # Edges within param1 dimension (second graph structure)
    assert result.has_edge(
        idx('param2', 'x', dims=('param1',)), idx('f', 'x', dims=('param1',))
    )
    assert result.has_edge(
        idx('param2', 'y', dims=('param1',)), idx('f', 'y', dims=('param1',))
    )
    assert result.has_edge(
        idx('param2', 'z', dims=('param1',)), idx('f', 'z', dims=('param1',))
    )

    # Edges from dim_0 to param1 grouping (first groupby)
    assert result.has_edge(idx('d', 0), idx('e', 'x', dims=('param1',)))
    assert result.has_edge(idx('d', 1), idx('e', 'x', dims=('param1',)))
    assert result.has_edge(idx('d', 2), idx('e', 'y', dims=('param1',)))
    assert result.has_edge(idx('d', 3), idx('e', 'z', dims=('param1',)))

    # Edges from param1 to param2 grouping (second groupby)
    assert result.has_edge(
        idx('f', 'x', dims=('param1',)), idx('grouped-f', 0, dims=('param2',))
    )
    assert result.has_edge(
        idx('f', 'y', dims=('param1',)), idx('grouped-f', 1, dims=('param2',))
    )
    assert result.has_edge(
        idx('f', 'z', dims=('param1',)), idx('grouped-f', 1, dims=('param2',))
    )


def test_group_in_different_ways() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('attach2', 'd')
    df = pd.DataFrame(
        {'a': [11, 22, 33], 'param1': ['a', 'a', 'b'], 'param2': ['x', 'y', 'x']}
    )

    graph = cb.Graph(g)
    mapped = graph.map(df)
    grouped = mapped.groupby('param1').reduce('b', name='grouped1')
    grouped2 = mapped.groupby('param2').reduce('b', name='grouped2')

    # Map helper node over param2 so we have a place where we can attached the grouping
    # by param2.
    grouped = grouped.map(
        pd.DataFrame({'attach2': [None, None], 'param2': ['x', 'y']}).set_index(
            'param2'
        )
    )
    grouped['attach2'] = grouped2['grouped2']

    # with pytest.raises(KeyError, match='dim_0'):
    grouped.to_networkx()
