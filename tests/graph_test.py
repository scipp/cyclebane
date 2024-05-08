# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipp as sc
import xarray as xr

import cyclebane as cb


@pytest.mark.parametrize('params', [{}, pd.DataFrame()])
def test_map_raises_when_mapping_over_empty(params) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    with pytest.raises(ValueError):
        graph.map(params)


@pytest.mark.parametrize(
    'params',
    [
        {'c': [1, 2]},
        {'a': [1, 2], 'c': [1, 2]},
        pd.DataFrame({'a': [1, 2], 'c': [1, 2]}),
    ],
)
def test_map_adds_node_when_mapping_nonexistent_node(params) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    result = mapped.to_networkx()
    c_data = [
        data
        for node, data in result.nodes(data=True)
        if getattr(node, 'name', None) == 'c'
    ]
    c_values = [data['value'] for data in c_data]
    assert c_values == [1, 2]


def test_map_raises_if_mapping_non_source_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    with pytest.raises(ValueError, match="Mapped node 'b' is not a source node"):
        graph.map({'b': [1, 2]})


def test_map_raises_if_mapping_previously_mapped_source_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2]})
    with pytest.raises(ValueError, match="Node 'a' has already been mapped"):
        mapped.map({'a': [1, 2]})


def test_map_raises_if_shapes_of_values_are_incompatible() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    with pytest.raises(
        ValueError, match="value sequences in a map operation must have the same shape"
    ):
        graph.map({'a': [1, 2], 'b': [1, 2, 3]})


def test_map_over_list_adds_value_attrs_to_source_nodes() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]
    b_data = [data for node, data in result.nodes(data=True) if node.name == 'b']
    assert not any('value' in data for data in b_data)


def test_map_does_not_duplicate_unrelated_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('x', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    result = mapped.to_networkx()
    assert len(result.nodes) == 3 + 3 + 1


def test_chained_map_over_list() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('x', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'x': [4, 5]})
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]

    x_data = [data for node, data in result.nodes(data=True) if node.name == 'x']
    x_values = [data['value'] for data in x_data]
    assert x_values == [4, 5]


def test_map_does_not_descent_into_nested_lists() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [[1, 2], [3, 4]]})
    assert len(mapped.to_networkx().nodes) == 2 + 2


def test_map_adds_axis_in_position_0_like_numpy_stack() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'b': [4, 5]})

    reduced = mapped.reduce('c', name='sum', axis=0)
    result = reduced.to_networkx()
    # Axis 0 should have length 2, so reducing it should leave us with 3 sink nodes,
    # i.e., the ones relating to the *first* call to map.
    sink_nodes = [node for node, degree in result.out_degree() if degree == 0]
    assert len(sink_nodes) == 3


def test_map_2d_numpy_array() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': np.array([[1, 2, 3], [4, 5, 6]])})
    assert len(mapped.to_networkx().nodes) == 3 * 2 * 2


def test_map_2d_xarray_dataarray() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    da = xr.DataArray(dims=('x', 'y'), data=[[1, 2, 3], [4, 5, 6]])
    print(da, da.dims, da.shape)
    mapped = graph.map({'a': da})
    assert len(mapped.to_networkx().nodes) == 3 * 2 * 2


def test_map_pandas_dataframe() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    result = mapped.to_networkx()
    assert len(result.nodes) == 3 * 3

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == params['a'].to_list()

    b_data = [data for node, data in result.nodes(data=True) if node.name == 'b']
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
    assert mapped.index_names == ('abcde',)


def test_map_pandas_dataframe_uses_index_values() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    params.index = [11, 22, 33]
    params.index.name = 'abcde'
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    for node in mapped.to_networkx().nodes:
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
    result = mapped.to_networkx()
    assert len(result.nodes) == 3 * 3

    int_data = [data for node, data in result.nodes(data=True) if node.name == int]
    int_values = [data['value'] for data in int_data]
    assert int_values == raw_params[int]

    float_data = [data for node, data in result.nodes(data=True) if node.name == float]
    float_values = [data['value'] for data in float_data]
    assert float_values == raw_params[float]


def test_map_scipp_variable() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    x = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
    mapped = graph.map({'a': x})
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values == list(x)


def test_map_2d_scipp_variable() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    values = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    mapped = graph.map({'a': values})
    assert mapped.index_names == ('x', 'y')
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
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
    assert len(mapped.to_networkx().nodes) == 2 + 2 + 2


def test_map_reduce() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('x', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'x': [4, 5]})
    reduced = mapped.reduce(name='func', axis=1)
    # Axis 0 reduces 'x', so there are 2 reduce nodes.
    assert len(reduced.to_networkx().nodes) == 19
    # Axis 1 reduces 'a', so there are 3 reduce nodes.
    reduced = mapped.reduce(name='func', axis=0)
    assert len(reduced.to_networkx().nodes) == 20

    a_data = [
        data
        for node, data in reduced.to_networkx().nodes(data=True)
        if node.name == 'a'
    ]
    a_values = [data['value'] for data in a_data]
    assert a_values == [1, 2, 3]


def test_reduce_all_axes() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]}).map({'b': [4, 5]})
    reduced = mapped.reduce(name='sum', attrs={'func': 'sum'})
    # No axis or index given, all axes are reduced, so the new node has no index part.
    assert 'sum' in reduced.graph
    assert reduced.graph.nodes['sum'] == {'func': 'sum'}


def test_reduce_preserves_reduced_index_names() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    graph = cb.Graph(g).map({'a': sc.ones(dims=['x', 'y'], shape=(2, 3))})
    reduced = graph.reduce('b', name='combine')
    # The new node is reduced, but the graph in its entirety still has the dims.
    assert reduced.index_names == ('x', 'y')


def test_reduce_raises_if_new_node_name_exists() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('other', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    with pytest.raises(ValueError):
        mapped.reduce(name='other')


@pytest.mark.parametrize('indexer', [{'axis': 1}, {'index': 'y'}])
def test_reduce_raises_if_axis_or_does_not_exist(indexer) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.arange('x', 3)})
    with pytest.raises(ValueError):
        mapped.reduce(name='combine', **indexer)


@pytest.mark.parametrize('indexer', [{'axis': 1}, {'index': 'y'}])
def test_reduce_raises_of_node_does_not_have_the_specified_axis_or_index(
    indexer,
) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.arange('x', 3)})
    with pytest.raises(ValueError):
        # We mapped 'a' but 'b' is not a descendant of 'a'.
        mapped.reduce('b', name='combine', **indexer)


def test_reduce_works_with_related_unmapped_nodes() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    reduced = mapped.reduce('c', name='combine')
    result = reduced.to_networkx()
    assert len(result.nodes) == 3 + 3 + 1 + 1  # 3a + 3b + 1c + 1reduce


@pytest.mark.parametrize('indexer', [{'axis': 0}, {'index': 'x'}])
def test_can_reduce_same_axis_or_index_on_multiple_nodes(indexer) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('a', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.ones(dims=['x'], shape=(3,))})
    reduced = (
        mapped.reduce('a', name='reduce-a', **indexer)
        .reduce('b', name='reduce-b', **indexer)
        .reduce('c', name='reduce-c', **indexer)
    )
    result = reduced.to_networkx()
    assert len(result.nodes) == 3 + 3 + 3 + 1 + 1 + 1


def test_can_reduce_same_node_multiple_times() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.ones(dims=['x'], shape=(3,))})
    reduced = mapped.reduce('b', name='c1', axis=0).reduce('b', name='c2', axis=0)
    result = reduced.to_networkx()
    assert len(result.nodes) == 3 + 3 + 1 + 1
    c1_parents = [n for n in result.predecessors('c1')]
    c2_parents = [n for n in result.predecessors('c2')]
    assert c1_parents == c2_parents


def test_can_reduce_different_axes_or_indices_of_same_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.ones(dims=['x', 'y'], shape=(3, 3))})
    reduced = (
        mapped.reduce('b', name='c0', axis=0)
        .reduce('b', name='c1', axis=1)
        .reduce('b', name='cx', index='x')
        .reduce('b', name='cy', index='y')
    )
    # Helper so we can get all the parents of the reduce nodes.
    helper = (
        reduced.reduce('c0', name='d0', axis=0)
        .reduce('c1', name='d1', axis=0)
        .reduce('cx', name='dx', index='y')
        .reduce('cy', name='dy', index='x')
    ).to_networkx()
    reduced = reduced.to_networkx()

    assert len(reduced.nodes) == 9 + 9 + 4 * 3
    c0s = [n for n in helper.predecessors('d0')]
    c1s = [n for n in helper.predecessors('d1')]
    cxs = [n for n in helper.predecessors('dx')]
    cys = [n for n in helper.predecessors('dy')]

    for c0, cx in zip(c0s, cxs):
        c0_parents = [n for n in reduced.predecessors(c0)]
        cx_parents = [n for n in reduced.predecessors(cx)]
        assert c0_parents == cx_parents

    for c1, cy in zip(c1s, cys):
        c1_parents = [n for n in reduced.predecessors(c1)]
        cy_parents = [n for n in reduced.predecessors(cy)]
        assert c1_parents == cy_parents


def test_axis_in_reduce_refers_to_node_axis_not_graph_axis() -> None:
    # TODO Is this actually the behavior we want?
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    graph = graph.map({'a': sc.ones(dims=['x', 'y', 'z'], shape=(2, 2, 2))})
    graph = graph.reduce('b', name='c', index='x')

    # Axis 1 of the graph is 'y', but we are reducing at 'c' which is (y, z)
    # so axis 1 of the reduce is 'z'.
    result = graph.reduce('c', name='d', axis=1).to_networkx()
    d_nodes = [n for n in result.nodes if n.name == 'd']
    # 'y' is left, even though axis 1 of the graph is 'y'.
    assert all(n.index.axes == ('y',) for n in d_nodes)


def test_setitem_raises_TypeError_if_given_networkx_graph() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    with pytest.raises(TypeError):
        graph['a'] = nx.DiGraph()


def test_setitem_with_other_graph_keeps_nodename_of_key_but_replaces_node_data() -> (
    None
):
    g1 = nx.DiGraph()
    g1.add_edge('b', 'a')
    g1.nodes['b']['attr'] = 1
    g2 = nx.DiGraph()
    g2.add_edge('d', 'c')
    g2.nodes['c']['attr'] = 2

    graph = cb.Graph(g1)
    graph['b'] = cb.Graph(g2)
    assert 'b' in graph.to_networkx()
    nx_graph = graph.to_networkx()
    assert set(nx_graph.nodes) == {'a', 'b', 'd'}
    assert len(nx_graph.edges) == 2
    assert nx_graph.has_edge('d', 'b')
    assert nx_graph.has_edge('b', 'a')
    assert nx_graph.nodes['b'] == {'attr': 2}


def test_setitem_raises_on_conflicting_ancestor_node_data() -> None:
    g1 = nx.DiGraph()
    g1.add_edge('a', 'b')
    g1.nodes['a']['attr'] = 1
    g1.add_edge('x', 'b')
    g2 = nx.DiGraph()
    g2.add_edge('a', 'x')
    g2.nodes['a']['attr'] = 2

    graph = cb.Graph(g1)
    with pytest.raises(ValueError, match="Node data differs for node 'a'"):
        graph['x'] = cb.Graph(g2)


def test_setitem_raises_on_conflicting_input_nodes_in_ancestor() -> None:
    g1 = nx.DiGraph()
    g1.add_edge('a1', 'b')
    g1.add_edge('b', 'c')
    g1.add_edge('x', 'c')
    g2 = nx.DiGraph()
    g2.add_edge('a2', 'b')
    g2.add_edge('b', 'x')

    graph = cb.Graph(g1)
    with pytest.raises(ValueError, match="Node inputs differ for node 'b'"):
        graph['x'] = cb.Graph(g2)


def test_getitem_keeps_only_relevant_indices() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    assert mapped['a'].indices == {'dim_0': range(3)}
    assert mapped['b'].indices == {}  # Not mapped
    assert mapped['c'].indices == {'dim_0': range(3)}
