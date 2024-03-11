# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import networkx as nx
import pytest
import scipp as sc

import cyclebane as cb


def test_map_raises_if_mapping_non_source_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    with pytest.raises(ValueError):
        graph.map({'b': [1, 2]})


def test_map_raises_if_mapping_previously_mapped_source_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2]})
    with pytest.raises(ValueError):
        mapped.map({'a': [1, 2]})


def test_map_raises_if_shapes_of_values_are_incompatible() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    with pytest.raises(ValueError):
        graph.map({'a': [1, 2], 'b': [1, 2, 3]})


def test_map_over_list() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('x', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'x': [4, 5]})

    a_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]

    x_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'x']
    x_values = [data['value'] for data in x_data]
    assert x_values == [4, 5]


def test_map_scipp_variable() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    x = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
    mapped = graph.map({'a': x})

    a_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == list(x)


def test_map_multiple_joint_index() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2], 'b': [4, 5]})
    # a and b have a common index, so we get 2 c's, not 4.
    assert len(mapped.graph.nodes) == 2 + 2 + 2


def test_map_reduce() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('x', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'x': [4, 5]})
    reduced = mapped.reduce('c', func='func', axis=1)
    # Axis 1 reduces 'x', so there are 3 reduce nodes.
    assert len(reduced.graph.nodes) == 20
    # Axis 0 reduces 'a', so there are 2 reduce nodes.
    reduced = mapped.reduce('c', func='func', axis=0)
    assert len(reduced.graph.nodes) == 19

    a_data = [data for node, data in reduced.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]
