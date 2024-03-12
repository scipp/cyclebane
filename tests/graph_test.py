# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import networkx as nx
import numpy as np
import pandas as pd
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


def test_map_does_not_descent_into_nested_lists() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [[1, 2], [3, 4]]})
    assert len(mapped.graph.nodes) == 2 + 2


def test_map_adds_axis_in_position_0_like_numpy_stack() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'b': [4, 5]})

    reduced = mapped.reduce('c', name='sum', axis=0)
    # Axis 0 should have length 2, so reducing it should leave us with 3 sink nodes,
    # i.e., the ones relating to the *first* call to map.
    sink_nodes = [node for node, degree in reduced.graph.out_degree() if degree == 0]
    assert len(sink_nodes) == 3


def test_map_2d_numpy_array() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': np.array([[1, 2, 3], [4, 5, 6]])})
    assert len(mapped.graph.nodes) == 3 * 2 * 2


def test_map_pandas_dataframe() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    assert len(mapped.graph.nodes) == 3 * 3

    a_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == params['a'].to_list()

    b_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'b']
    b_values = [data['value'] for data in b_data]
    assert b_values == params['b'].to_list()


def test_map_pandas_dataframe_uses_index_name() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    params.index.name = 'abcde'
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    assert mapped.index_names == {'abcde'}


def test_map_pandas_dataframe_uses_index_values() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    params.index = [11, 22, 33]
    params.index.name = 'abcde'
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    for node in mapped.graph.nodes:
        assert node.index.axes == ('abcde',)
        assert node.index.values[0] in [11, 22, 33]


def test_map_pandas_dataframe_with_type_as_col_name_works() -> None:
    # This is a slightly obscure way of naming columns, but it is used by Sciline.
    # There is a special test for it since DataFrame.__getitem__ does not work with
    # this kind of column name.
    raw_params = {int: [1, 2, 3], float: [0.1, 0.2, 0.3]}
    params = pd.DataFrame(raw_params)
    g = nx.DiGraph()
    g.add_edge(int, 'a')
    g.add_edge(float, 'a')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    assert len(mapped.graph.nodes) == 3 * 3

    int_data = [
        data for node, data in mapped.graph.nodes(data=True) if node.name == int
    ]
    int_values = [data['value'] for data in int_data]
    assert int_values == raw_params[int]

    float_data = [
        data for node, data in mapped.graph.nodes(data=True) if node.name == float
    ]
    float_values = [data['value'] for data in float_data]
    assert float_values == raw_params[float]


def test_map_scipp_variable() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    x = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
    mapped = graph.map({'a': x})

    a_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == list(x)


def test_map_2d_scipp_variable() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    values = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    mapped = graph.map({'a': values})

    a_data = [data for node, data in mapped.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values[0:3] == list(values['x', 0])
    assert a_values[3:6] == list(values['x', 1])


def test_reduce_scipp_mapped() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    x = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
    mapped = graph.map({'a': x})
    reduced = mapped.reduce('b', name='sum', index='x')

    assert 'sum' in reduced.graph


def test_map_with_previously_mapped_index_name_raises() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    values = sc.arange('x', 3)

    mapped = graph.map({'a': values})
    with pytest.raises(ValueError):
        mapped.map({'b': values})


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
    reduced = mapped.reduce('c', name='func', axis=1)
    # Axis 0 reduces 'x', so there are 2 reduce nodes.
    assert len(reduced.graph.nodes) == 19
    # Axis 1 reduces 'a', so there are 3 reduce nodes.
    reduced = mapped.reduce('c', name='func', axis=0)
    assert len(reduced.graph.nodes) == 20

    a_data = [data for node, data in reduced.graph.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]


def test_reduce_all_axes() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'b': [4, 5]})
    reduced = mapped.reduce('c', name='sum', attrs={'func': 'sum'})
    # No axis or index given, all axes are reduced, so the new node has no index part.
    assert 'sum' in reduced.graph
    assert reduced.graph.nodes['sum'] == {'func': 'sum'}
