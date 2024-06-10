# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Hashable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipp as sc
import xarray as xr

import cyclebane as cb


def idx(
    name: str, *index: int, offset=None, dims: tuple[str, ...] = ('dim_0', 'dim_1')
) -> cb.graph.NodeName:
    """Helper to create a NodeName with a tuple of indices."""
    return cb.graph.NodeName(
        name,
        cb.graph.IndexValues(dims[offset : len(index) + (offset or 0)], tuple(index)),
    )


@pytest.mark.parametrize('params', [{}, pd.DataFrame()])
def test_map_can_map_over_empty(params) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    assert len(mapped.to_networkx().nodes) == 2


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
    assert result.nodes[idx('c', 0)] == {'value': 1}
    assert result.nodes[idx('c', 1)] == {'value': 2}


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

    assert result.nodes[idx('a', 0)] == {'value': 1}
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}
    assert result.nodes[idx('b', 0)] == {}
    assert result.nodes[idx('b', 1)] == {}
    assert result.nodes[idx('b', 2)] == {}


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

    assert result.nodes[idx('a', 0)] == {'value': 1}
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}
    assert result.nodes[idx('x', 0, offset=1)] == {'value': 4}
    assert result.nodes[idx('x', 1, offset=1)] == {'value': 5}


def test_map_does_not_descend_into_nested_lists() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [[1, 2], [3, 4]]})
    assert mapped.index_names == ('dim_0',)
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


def test_map_2d_numpy_array_sets_up_default_index_names() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': np.array([[1, 2, 3], [4, 5, 6]])})
    assert mapped.index_names == ('dim_0', 'dim_1')
    assert len(mapped.to_networkx().nodes) == 3 * 2 * 2


def test_map_2d_xarray_dataarray_uses_dims_as_index_names() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    da = xr.DataArray(dims=('x', 'y'), data=[[1, 2, 3], [4, 5, 6]])
    mapped = graph.map({'a': da})
    assert mapped.index_names == ('x', 'y')
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

    assert result.nodes[idx('a', 0)] == {'value': 1}
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}

    assert result.nodes[idx('b', 0)] == {'value': 4}
    assert result.nodes[idx('b', 1)] == {'value': 5}
    assert result.nodes[idx('b', 2)] == {'value': 6}


def test_map_pandas_dataframe_sets_up_default_index_name() -> None:
    params = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(params)
    assert mapped.index_names == ('dim_0',)


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

    assert result.nodes[idx(int, 0)] == {'value': 1}
    assert result.nodes[idx(int, 1)] == {'value': 2}
    assert result.nodes[idx(int, 2)] == {'value': 3}

    assert result.nodes[idx(float, 0)] == {'value': 0.1}
    assert result.nodes[idx(float, 1)] == {'value': 0.2}
    assert result.nodes[idx(float, 2)] == {'value': 0.3}


def test_map_scipp_variable_uses_dims_as_index_names() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    x = sc.array(dims=['x'], values=[1, 2, 3], unit='m')
    mapped = graph.map({'a': x})
    assert mapped.index_names == ('x',)
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = sc.concat([data['value'] for data in a_data], 'x')
    assert sc.identical(a_values.data, x)


def test_map_2d_scipp_variable_uses_dims_as_index_names() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    values = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    mapped = graph.map({'a': values})
    assert mapped.index_names == ('x', 'y')
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = sc.concat([data['value'] for data in a_data], 'y')
    assert sc.identical(a_values[0:3].data, values['x', 0])
    assert sc.identical(a_values[3:6].data, values['x', 1])


def test_map_scipp_data_array_uses_coord_as_indices_if_present() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    values = sc.array(dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]], unit='m')
    da = sc.DataArray(
        values, coords={'y': sc.linspace(dim='y', start=0, stop=1, num=3, unit='mm')}
    )
    mapped = graph.map({'a': da})
    assert mapped.index_names == ('x', 'y')
    result = mapped.to_networkx()

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = sc.concat([data['value'] for data in a_data], 'y')
    del a_values.coords['x']  # auto range index not in reference
    assert sc.identical(a_values[0:3], da['x', 0])
    assert sc.identical(a_values[3:6], da['x', 1])

    for name in ['a', 'b']:
        # Note that due to current shortcomings in scipp the indices are tuples of the
        # form (value, unit) and not the corresponding 0-D scipp.Variable.
        assert idx(name, 0, (0.0, 'mm'), dims=('x', 'y')) in result
        assert idx(name, 0, (0.5, 'mm'), dims=('x', 'y')) in result
        assert idx(name, 0, (1.0, 'mm'), dims=('x', 'y')) in result
        assert idx(name, 1, (0.0, 'mm'), dims=('x', 'y')) in result
        assert idx(name, 1, (0.5, 'mm'), dims=('x', 'y')) in result
        assert idx(name, 1, (1.0, 'mm'), dims=('x', 'y')) in result


def test_map_xarray_data_array_uses_coord_as_indices_if_present() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    da = xr.DataArray(
        dims=('x', 'y'),
        data=np.array([[1, 2, 3], [4, 5, 6]]),
        coords={'y': np.linspace(start=0, stop=1, num=3)},
    )
    mapped = graph.map({'a': da})
    assert mapped.index_names == ('x', 'y')
    result = mapped.to_networkx()

    for name in ['a', 'b']:
        assert idx(name, 0, 0.0, dims=('x', 'y')) in result
        assert idx(name, 0, 0.5, dims=('x', 'y')) in result
        assert idx(name, 0, 1.0, dims=('x', 'y')) in result
        assert idx(name, 1, 0.0, dims=('x', 'y')) in result
        assert idx(name, 1, 0.5, dims=('x', 'y')) in result
        assert idx(name, 1, 1.0, dims=('x', 'y')) in result


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
    with pytest.raises(ValueError, match="Conflicting new index names"):
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
    result = reduced.to_networkx()
    assert len(result.nodes) == 20

    assert result.nodes[idx('a', 0)] == {'value': 1}
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}
    assert result.nodes[idx('x', 0, offset=1)] == {'value': 4}
    assert result.nodes[idx('x', 1, offset=1)] == {'value': 5}


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
    with pytest.raises(ValueError, match="Node 'other' already exists in the graph."):
        mapped.reduce(name='other')


@pytest.mark.parametrize('indexer', [{'axis': 1}, {'index': 'y'}])
def test_reduce_raises_if_axis_or_does_not_exist(indexer) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.arange('x', 3)})
    with pytest.raises(ValueError, match="does not have"):
        mapped.reduce(name='combine', **indexer)


@pytest.mark.parametrize('indexer', [{'axis': 1}, {'index': 'y'}])
def test_reduce_raises_if_node_does_not_have_the_specified_axis_or_index(
    indexer,
) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': sc.arange('x', 3)})
    with pytest.raises(ValueError, match="Node 'b' does not have "):
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
    c1_parents = list(result.predecessors('c1'))
    c2_parents = list(result.predecessors('c2'))
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
    c0s = list(helper.predecessors('d0'))
    c1s = list(helper.predecessors('d1'))
    cxs = list(helper.predecessors('dx'))
    cys = list(helper.predecessors('dy'))

    for c0, cx in zip(c0s, cxs, strict=True):
        c0_parents = list(reduced.predecessors(c0))
        cx_parents = list(reduced.predecessors(cx))
        assert c0_parents == cx_parents

    for c1, cy in zip(c1s, cys, strict=True):
        c1_parents = list(reduced.predecessors(c1))
        cy_parents = list(reduced.predecessors(cy))
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


def test_setitem_replaces_nodes_that_are_not_ancestors_of_unrelated_node() -> None:
    g1 = nx.DiGraph()
    g1.add_edge('a', 'b')
    g1.add_edge('b', 'c')
    g1.add_edge('c', 'd')
    graph = cb.Graph(g1)
    g2 = nx.DiGraph()
    g2.add_edge('b', 'c')
    graph['c'] = cb.Graph(g2)
    assert 'a' not in graph.to_networkx()


def test_setitem_preserves_nodes_that_are_ancestors_of_unrelated_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('b', 'd')
    g.add_edge('c', 'd')
    graph = cb.Graph(g)
    graph['c'] = graph['c']
    nx.utils.graphs_equal(graph.to_networkx(), g)


def test_getitem_returns_graph_containing_only_key_and_ancestors() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('c', 'd')
    g.add_edge('x', 'd')

    graph = cb.Graph(g)
    subgraph = graph['c']
    result = subgraph.to_networkx()
    assert len(result.nodes) == 3
    assert len(result.edges) == 2
    assert 'a' in result
    assert 'b' in result
    assert 'c' in result


def test_getitem_setitem_with_no_effects() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.nodes['a']['value'] = 1

    graph = cb.Graph(g)
    graph['b'] = graph['b']
    assert graph.to_networkx().nodes['a']['value'] == 1
    # Note that we cannot currently do the following:
    #   mapped = graph.map({'a': [1, 2, 3]})
    #   mapped['b'] = mapped['b']
    # because the check for compatible indices/node-values is not implemented.


def test_getitem_keeps_only_relevant_indices() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    assert mapped['a'].indices == {'dim_0': range(3)}
    assert mapped['b'].indices == {}  # Not mapped
    assert mapped['c'].indices == {'dim_0': range(3)}


def test_getitem_keeps_only_relevant_node_values() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    # This fails due to existing mapping...
    with pytest.raises(ValueError, match='has already been mapped'):
        mapped.map({'a': [1, 2]})
    # ... but getitem drops the 'a' mapping, so we can map 'a' again:
    mapped['b'].map({'a': [1, 2]})


def test_getitem_on_mapped_graph_with_base_node_name_returns_mapped_node() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    result = mapped['b'].to_networkx()
    assert len(result.nodes) == 6


def test_setitem_with_mapped_sink_node_raises_if_target_is_not_mapped() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    b = cb.Graph(nx.DiGraph()).map({'b': [11, 12]})
    with pytest.raises(
        NotImplementedError, match="Mapped nodes not supported yet in __setitem__"
    ):
        graph['b'] = b


def test_setitem_with_mapped_operands_raises_on_conflict() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    d = cb.Graph(nx.DiGraph()).map({'b': [11, 12]}).reduce('b', name='d')
    with pytest.raises(ValueError, match="Conflicting new index names"):
        mapped['x'] = d


def test_setitem_currently_does_not_allow_compatible_indices() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('c', 'd')

    graph = cb.Graph(g)
    mapped1 = graph.map({'a': [1, 2, 3]})
    mapped2 = graph['d'].map({'c': [11, 12, 13]}).reduce('d', name='e')
    # Note: This is a limitation of the current implementation. We could check if the
    # indices are identical and allow this. For simplicity we currently do not.
    with pytest.raises(ValueError, match="Conflicting new index names"):
        mapped1['x'] = mapped2


@pytest.mark.parametrize(
    'node_values',
    [
        {'a': [1, 2, 3]},
        {'a': [1, 2, 3], 'b': [11, 12, 13]},
        {'a': np.array([1, 2, 3])},
        {'a': np.array([1, 2, 3]), 'b': np.array([11, 12, 13])},
        pd.DataFrame({'a': [1, 2, 3]}),
        pd.DataFrame({'a': [1, 2, 3], 'b': [11, 12, 13]}),
        {'a': sc.array(dims=['x'], values=[1, 2, 3])},
        {
            'a': sc.array(dims=['x'], values=[1, 2, 3]),
            'b': sc.array(dims=['x'], values=[11, 12, 13]),
        },
        {'a': xr.DataArray(dims=('x',), data=[1, 2, 3])},
        {
            'a': xr.DataArray(dims=('x',), data=[1, 2, 3]),
            'b': xr.DataArray(dims=('x',), data=[11, 12, 13]),
        },
    ],
)
def test_setitem_allows_compatible_node_values(node_values) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'c')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    mapped = graph.map(node_values).reduce('c', name='d')
    mapped['x'] = mapped['d']
    assert len(mapped.index_names) == 1


def test_setitem_raises_if_node_values_equivalent_but_of_different_type() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    graph = cb.Graph(g)
    mapped1 = graph.map({'a': [1, 2]}).reduce('b', name='d')
    mapped2 = graph.map({'a': np.array([1, 2])}).reduce('b', name='d')
    # One could imagine treating this as equivalent, but we are strict in the
    # comparison.
    with pytest.raises(ValueError, match="Conflicting new index names"):
        mapped1['x'] = mapped2['d']


def test_setitem_raises_if_node_values_incompatible() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    graph = cb.Graph(g)
    mapped1 = graph.map({'a': [1, 2]}).reduce('b', name='d')
    mapped2 = graph.map({'a': sc.array(dims=('x',), values=[1, 2])}).reduce(
        'b', name='d'
    )
    with pytest.raises(ValueError, match="has already been mapped"):
        mapped1['x'] = mapped2['d']


def test_setitem_does_currently_not_support_slice_assignment() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')

    graph = cb.Graph(g)
    with pytest.raises(NotImplementedError):
        graph['b':'b'] = graph['b']
    with pytest.raises(NotImplementedError):
        graph['b':'a'] = graph['b']


def test_setitem_raises_if_value_graph_does_not_have_unique_sink() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('a', 'c')

    graph = cb.Graph(g)
    with pytest.raises(ValueError, match="Graph must have exactly one sink node"):
        graph['a'] = graph


@pytest.mark.parametrize(
    'param_table',
    [{'a': [1, 2, 3]}, {'a': np.array([1, 2, 3])}, pd.DataFrame({'a': [1, 2, 3]})],
)
def test_slice_by_position(param_table: Mapping[Hashable, Sequence[int]]) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map(param_table)
    sliced = mapped.by_position('dim_0')[1:3]
    result = sliced.to_networkx()
    assert cb.graph.NodeName('a', cb.graph.IndexValues(('dim_0',), (0,))) not in result
    assert cb.graph.NodeName('a', cb.graph.IndexValues(('dim_0',), (1,))) in result
    assert cb.graph.NodeName('a', cb.graph.IndexValues(('dim_0',), (2,))) in result

    assert idx('a', 0) not in result
    assert result.nodes[idx('a', 1)] == {'value': 2}
    assert result.nodes[idx('a', 2)] == {'value': 3}


@pytest.mark.parametrize(
    'values',
    [
        np.array([[1, 2, 3], [4, 5, 6]]),
        xr.DataArray(dims=('dim_0', 'dim_1'), data=[[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_by_position_2d_slice_outer(values) -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': values})
    sliced = mapped.by_position('dim_0')[1:]
    result = sliced.to_networkx()

    assert idx('a', 0, 0) not in result
    assert idx('a', 0, 1) not in result
    assert idx('a', 0, 2) not in result
    assert idx('a', 1, 0) in result
    assert idx('a', 1, 1) in result
    assert idx('a', 1, 2) in result

    a_data = [data for node, data in result.nodes(data=True) if node.name == 'a']
    a_values = [data['value'] for data in a_data]
    assert a_values[0:3] == [4, 5, 6]


def test_by_position_2d_slice_inner() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')

    graph = cb.Graph(g)
    mapped = graph.map({'a': np.array([[1, 2, 3], [4, 5, 6]])})
    sliced = mapped.by_position('dim_1')[:2]
    result = sliced.to_networkx()

    assert idx('a', 0, 0) in result
    assert idx('a', 0, 1) in result
    assert idx('a', 0, 2) not in result
    assert idx('a', 1, 0) in result
    assert idx('a', 1, 1) in result
    assert idx('a', 1, 2) not in result

    assert result.nodes[idx('a', 0, 0)] == {'value': 1}
    assert result.nodes[idx('a', 0, 1)] == {'value': 2}
    assert result.nodes[idx('a', 1, 0)] == {'value': 4}
    assert result.nodes[idx('a', 1, 1)] == {'value': 5}


def test_node_attrs_are_preserved() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.nodes['a']['attr'] = 1

    graph = cb.Graph(g)
    result = graph.to_networkx()
    assert result.nodes['a'] == {'attr': 1}


def test_node_attrs_are_preserved_in_getitem() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.nodes['a']['attr1'] = 1
    g.nodes['b']['attr2'] = 2
    g.nodes['c']['attr3'] = 3

    graph = cb.Graph(g)
    result = graph['c'].to_networkx()
    assert result.nodes['a'] == {'attr1': 1}
    assert result.nodes['b'] == {'attr2': 2}


def test_node_attrs_are_preserved_in_setitem() -> None:
    g1 = nx.DiGraph()
    g1.add_edge('a', 'c')
    g1.add_edge('b', 'c')
    g1.nodes['a']['attr1'] = 1
    g1.nodes['b']['attr2'] = 2
    g1.nodes['c']['attr3'] = 3

    graph = cb.Graph(g1)
    g2 = nx.DiGraph()
    g2.add_edge('x', 'b')
    g2.nodes['x']['attr4'] = 4
    g2.nodes['b']['attr5'] = 5

    graph['b'] = cb.Graph(g2)

    result = graph.to_networkx()
    assert result.nodes['a'] == {'attr1': 1}
    assert result.nodes['b'] == {'attr5': 5}  # b was replaced
    assert result.nodes['c'] == {'attr3': 3}
    assert result.nodes['x'] == {'attr4': 4}


def test_node_attrs_are_preserved_in_map() -> None:
    g = nx.DiGraph()
    g.add_edge('a', 'b')
    g.nodes['a']['attr'] = 11
    g.nodes['b']['attr'] = 22

    graph = cb.Graph(g)
    mapped = graph.map({'a': [1, 2, 3]})
    value_attr = 'myvalue'
    result = mapped.to_networkx(value_attr=value_attr)

    assert result.nodes[idx('a', 0)] == {'attr': 11, 'myvalue': 1}
    assert result.nodes[idx('a', 1)] == {'attr': 11, 'myvalue': 2}
    assert result.nodes[idx('a', 2)] == {'attr': 11, 'myvalue': 3}
    assert result.nodes[idx('b', 0)] == {'attr': 22}
    assert result.nodes[idx('b', 1)] == {'attr': 22}
    assert result.nodes[idx('b', 2)] == {'attr': 22}
