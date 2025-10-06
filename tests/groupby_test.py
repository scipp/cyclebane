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


class TestGroupbyBasicFunctionality:
    """Tests for basic groupby functionality with different group configurations."""

    def test_groupby_with_different_number_of_groups(self) -> None:
        """Test groupby with 2, 3, and many groups."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        # Test with 3 groups
        df = pd.DataFrame(
            {'a': [1, 2, 3, 4, 5, 6], 'param': ['x', 'x', 'y', 'y', 'z', 'z']}
        )
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        assert result.nodes[idx('c', 'x', dims=('param',))] == {}
        assert result.nodes[idx('c', 'y', dims=('param',))] == {}
        assert result.nodes[idx('c', 'z', dims=('param',))] == {}

        # Verify correct grouping
        assert result.has_edge(idx('b', 0), idx('c', 'x', dims=('param',)))
        assert result.has_edge(idx('b', 1), idx('c', 'x', dims=('param',)))
        assert result.has_edge(idx('b', 2), idx('c', 'y', dims=('param',)))
        assert result.has_edge(idx('b', 3), idx('c', 'y', dims=('param',)))
        assert result.has_edge(idx('b', 4), idx('c', 'z', dims=('param',)))
        assert result.has_edge(idx('b', 5), idx('c', 'z', dims=('param',)))

    def test_groupby_single_group(self) -> None:
        """Test groupby where all values belong to the same group."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame(
            {'a': [1, 2, 3, 4], 'param': ['same', 'same', 'same', 'same']}
        )
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        # Should have only one group
        assert result.nodes[idx('c', 'same', dims=('param',))] == {}

        # All b nodes should connect to the single group
        assert result.has_edge(idx('b', 0), idx('c', 'same', dims=('param',)))
        assert result.has_edge(idx('b', 1), idx('c', 'same', dims=('param',)))
        assert result.has_edge(idx('b', 2), idx('c', 'same', dims=('param',)))
        assert result.has_edge(idx('b', 3), idx('c', 'same', dims=('param',)))

    def test_groupby_single_element_per_group(self) -> None:
        """Test groupby where each group has only one element."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [1, 2, 3], 'param': ['x', 'y', 'z']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        # Should have three groups, each with one element
        assert result.nodes[idx('c', 'x', dims=('param',))] == {}
        assert result.nodes[idx('c', 'y', dims=('param',))] == {}
        assert result.nodes[idx('c', 'z', dims=('param',))] == {}

        # Each b node connects to its own group
        assert result.has_edge(idx('b', 0), idx('c', 'x', dims=('param',)))
        assert result.has_edge(idx('b', 1), idx('c', 'y', dims=('param',)))
        assert result.has_edge(idx('b', 2), idx('c', 'z', dims=('param',)))


class TestGroupbyDataTypes:
    """Tests for groupby with different data types."""

    def test_groupby_with_integer_groups(self) -> None:
        """Test groupby with integer group values."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [10, 20, 30, 40], 'group': [0, 0, 1, 1]})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('group').reduce('b', name='c')
        result = grouped.to_networkx()

        assert result.nodes[idx('c', 0, dims=('group',))] == {}
        assert result.nodes[idx('c', 1, dims=('group',))] == {}

        assert result.has_edge(idx('b', 0), idx('c', 0, dims=('group',)))
        assert result.has_edge(idx('b', 1), idx('c', 0, dims=('group',)))
        assert result.has_edge(idx('b', 2), idx('c', 1, dims=('group',)))
        assert result.has_edge(idx('b', 3), idx('c', 1, dims=('group',)))

    def test_groupby_with_float_groups(self) -> None:
        """Test groupby with float group values."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [10, 20, 30], 'group': [1.5, 1.5, 2.5]})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('group').reduce('b', name='c')
        result = grouped.to_networkx()

        assert result.nodes[idx('c', 1.5, dims=('group',))] == {}
        assert result.nodes[idx('c', 2.5, dims=('group',))] == {}

        assert result.has_edge(idx('b', 0), idx('c', 1.5, dims=('group',)))
        assert result.has_edge(idx('b', 1), idx('c', 1.5, dims=('group',)))
        assert result.has_edge(idx('b', 2), idx('c', 2.5, dims=('group',)))

    def test_groupby_with_named_index(self) -> None:
        """Test groupby with a Pandas Series that has a named index."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [10, 20, 30], 'param': ['x', 'x', 'y']})
        df.index.name = 'my_index'
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        # Index name should be used
        assert idx('a', 0, dims=('my_index',)) in result.nodes
        assert idx('c', 'x', dims=('param',)) in result.nodes

    def test_groupby_error_with_non_pandas_type(self) -> None:
        """Test that groupby raises NotImplementedError for non-Pandas types."""
        import pytest

        g = nx.DiGraph()
        g.add_edge('a', 'b')

        # Try with a list (SequenceAdapter)
        graph = cb.Graph(g).map({'a': [1, 2, 3]})

        # Error is raised in groupby() call itself, not reduce()
        with pytest.raises(NotImplementedError, match='only implemented for Pandas'):
            graph.groupby('a')


class TestGroupbyGraphStructure:
    """Tests for groupby with different graph structures."""

    def test_groupby_with_multiple_predecessors(self) -> None:
        """Test groupby on a node with multiple predecessors."""
        g = nx.DiGraph()
        g.add_edge('a', 'c')
        g.add_edge('b', 'c')

        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'param': ['x', 'x', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('c', name='d')
        result = grouped.to_networkx()

        # Both predecessors should have their nodes
        assert idx('a', 0) in result.nodes
        assert idx('b', 0) in result.nodes

        # Grouped nodes should exist
        assert idx('d', 'x', dims=('param',)) in result.nodes
        assert idx('d', 'y', dims=('param',)) in result.nodes

        # Edges from both predecessors to c
        assert result.has_edge(idx('a', 0), idx('c', 0))
        assert result.has_edge(idx('b', 0), idx('c', 0))

    def test_groupby_on_intermediate_node(self) -> None:
        """Test groupby on an intermediate node (not immediately after map)."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        df = pd.DataFrame({'a': [1, 2, 3, 4], 'param': ['x', 'x', 'y', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('c', name='d')
        result = grouped.to_networkx()

        # All intermediate nodes should exist
        assert idx('a', 0) in result.nodes
        assert idx('b', 0) in result.nodes
        assert idx('c', 0) in result.nodes

        # Chain should be preserved
        assert result.has_edge(idx('a', 0), idx('b', 0))
        assert result.has_edge(idx('b', 0), idx('c', 0))

        # Grouped nodes
        assert result.has_edge(idx('c', 0), idx('d', 'x', dims=('param',)))
        assert result.has_edge(idx('c', 1), idx('d', 'x', dims=('param',)))

    def test_groupby_with_attrs(self) -> None:
        """Test that attrs are correctly set on reduce nodes."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [1, 2, 3], 'param': ['x', 'x', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce(
            'b', name='c', attrs={'custom': 'value'}
        )
        result = grouped.to_networkx()

        # Attrs should be set on the reduce nodes
        assert result.nodes[idx('c', 'x', dims=('param',))]['custom'] == 'value'
        assert result.nodes[idx('c', 'y', dims=('param',))]['custom'] == 'value'


class TestGroupbyIntegration:
    """Tests for groupby integrated with other operations."""

    def test_groupby_combined_with_regular_reduce(self) -> None:
        """Test groupby combined with regular reduce in the same graph."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('c', 'd')

        df = pd.DataFrame({'a': [1, 2, 3], 'c': [10, 20, 30], 'param': ['x', 'x', 'y']})
        graph = cb.Graph(g).map(df)

        # Regular reduce on one branch
        reduced = graph.reduce('b', name='b_reduced')

        # Groupby reduce on another branch
        grouped = reduced.groupby('param').reduce('d', name='d_grouped')
        result = grouped.to_networkx()

        # Regular reduce node should exist (single node, no index)
        assert 'b_reduced' in result.nodes

        # Grouped reduce nodes should exist
        assert idx('d_grouped', 'x', dims=('param',)) in result.nodes
        assert idx('d_grouped', 'y', dims=('param',)) in result.nodes

    def test_groupby_with_branch_operations(self) -> None:
        """Test groupby combined with branch selection."""
        g1 = nx.DiGraph()
        g1.add_edge('a', 'b')

        g2 = nx.DiGraph()
        g2.add_edge('c', 'd')

        df = pd.DataFrame({'a': [1, 2, 3], 'c': [10, 20, 30], 'param': ['x', 'x', 'y']})
        graph1 = cb.Graph(g1).map(df)
        graph2 = cb.Graph(g2).map(df)

        # Combine graphs
        graph1['c'] = graph2['d']

        # Groupby on combined graph
        grouped = graph1.groupby('param').reduce('b', name='reduced')
        result = grouped.to_networkx()

        assert idx('reduced', 'x', dims=('param',)) in result.nodes
        assert idx('reduced', 'y', dims=('param',)) in result.nodes


class TestGroupbyEdgeCases:
    """Tests for groupby edge cases and error handling."""

    def test_groupby_on_nonexistent_node(self) -> None:
        """Test that groupby raises KeyError for non-existent node."""
        import pytest

        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [1, 2, 3]})
        graph = cb.Graph(g).map(df)

        with pytest.raises(KeyError):
            graph.groupby('nonexistent')

    def test_groupby_reduce_with_name_conflict(self) -> None:
        """Test that reduce after groupby raises error for existing node name."""
        import pytest

        g = nx.DiGraph()
        g.add_edge('a', 'b')

        df = pd.DataFrame({'a': [1, 2, 3], 'param': ['x', 'x', 'y']})
        graph = cb.Graph(g).map(df)

        # Error message says "already been mapped" not "already exists"
        with pytest.raises(ValueError, match="already been mapped"):
            graph.groupby('param').reduce('b', name='a')

    def test_groupby_with_uneven_group_sizes(self) -> None:
        """Test groupby where groups have very different sizes."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')

        # One group with 5 elements, another with 1
        df = pd.DataFrame(
            {'a': [1, 2, 3, 4, 5, 6], 'param': ['x', 'x', 'x', 'x', 'x', 'y']}
        )
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        # Both groups should exist
        assert idx('c', 'x', dims=('param',)) in result.nodes
        assert idx('c', 'y', dims=('param',)) in result.nodes

        # Check edges - group 'x' should have 5 incoming edges
        edges_to_x = [
            edge for edge in result.edges if edge[1] == idx('c', 'x', dims=('param',))
        ]
        assert len(edges_to_x) == 5

        # Group 'y' should have 1 incoming edge
        edges_to_y = [
            edge for edge in result.edges if edge[1] == idx('c', 'y', dims=('param',))
        ]
        assert len(edges_to_y) == 1

    def test_groupby_preserves_node_values(self) -> None:
        """Test that groupby preserves node values from original mapping."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('param', 'b')

        df = pd.DataFrame({'a': [11, 22, 33], 'param': ['x', 'x', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('b', name='c')
        result = grouped.to_networkx()

        # Original node values should be preserved
        assert result.nodes[idx('a', 0)]['value'] == 11
        assert result.nodes[idx('a', 1)]['value'] == 22
        assert result.nodes[idx('a', 2)]['value'] == 33
        assert result.nodes[idx('param', 0)]['value'] == 'x'
        assert result.nodes[idx('param', 1)]['value'] == 'x'
        assert result.nodes[idx('param', 2)]['value'] == 'y'


class TestGroupbyComplexScenarios:
    """Tests for complex groupby scenarios."""

    def test_groupby_diamond_pattern(self) -> None:
        """Test groupby in a diamond-shaped graph."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('a', 'c')
        g.add_edge('b', 'd')
        g.add_edge('c', 'd')

        df = pd.DataFrame({'a': [1, 2, 3, 4], 'param': ['x', 'x', 'y', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('d', name='e')
        result = grouped.to_networkx()

        # Both intermediate nodes should exist
        assert idx('b', 0) in result.nodes
        assert idx('c', 0) in result.nodes

        # Grouped nodes
        assert idx('e', 'x', dims=('param',)) in result.nodes
        assert idx('e', 'y', dims=('param',)) in result.nodes

        # Diamond structure should be preserved
        assert result.has_edge(idx('a', 0), idx('b', 0))
        assert result.has_edge(idx('a', 0), idx('c', 0))
        assert result.has_edge(idx('b', 0), idx('d', 0))
        assert result.has_edge(idx('c', 0), idx('d', 0))

    def test_groupby_linear_chain(self) -> None:
        """Test groupby on a long linear chain."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')
        g.add_edge('c', 'd')
        g.add_edge('d', 'e')

        df = pd.DataFrame({'a': [1, 2, 3, 4], 'param': ['x', 'x', 'y', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('e', name='f')
        result = grouped.to_networkx()

        # All intermediate nodes should exist
        assert idx('a', 0) in result.nodes
        assert idx('b', 0) in result.nodes
        assert idx('c', 0) in result.nodes
        assert idx('d', 0) in result.nodes
        assert idx('e', 0) in result.nodes

        # Chain should be preserved
        assert result.has_edge(idx('a', 0), idx('b', 0))
        assert result.has_edge(idx('b', 0), idx('c', 0))
        assert result.has_edge(idx('c', 0), idx('d', 0))
        assert result.has_edge(idx('d', 0), idx('e', 0))

    def test_groupby_with_multiple_source_nodes(self) -> None:
        """Test groupby where graph has multiple source nodes."""
        g = nx.DiGraph()
        g.add_edge('a', 'c')
        g.add_edge('b', 'c')
        # Note: both 'a' and 'b' are source nodes

        df = pd.DataFrame({'a': [1, 2], 'b': [10, 20], 'param': ['x', 'y']})
        graph = cb.Graph(g).map(df)
        grouped = graph.groupby('param').reduce('c', name='d')
        result = grouped.to_networkx()

        # Both source nodes should exist
        assert idx('a', 0) in result.nodes
        assert idx('b', 0) in result.nodes

        # Grouped nodes
        assert idx('d', 'x', dims=('param',)) in result.nodes
        assert idx('d', 'y', dims=('param',)) in result.nodes


class TestGroupbyChainedOperations:
    """Tests for chained groupby and other operations."""

    def test_three_groupby_operations_in_sequence(self) -> None:
        """Test three consecutive groupby operations."""
        g1 = nx.DiGraph()
        g1.add_edge('a', 'b')

        g2 = nx.DiGraph()
        g2.add_edge('c', 'd')

        g3 = nx.DiGraph()
        g3.add_edge('e', 'f')

        # First groupby
        grouped1 = (
            cb.Graph(g1)
            .map(pd.DataFrame({'a': [1, 2, 3, 4], 'p1': ['x', 'x', 'y', 'y']}))
            .groupby('p1')
            .reduce('b', name='gb1')
        )

        # Second groupby - use branch selection to get single sink
        mapped2 = cb.Graph(g2).map(
            pd.DataFrame({'c': [10, 20], 'p1': ['x', 'y'], 'p2': [0, 1]}).set_index(
                'p1'
            )
        )
        mapped2['c'] = grouped1['gb1']
        grouped2 = mapped2.groupby('p2').reduce('d', name='gb2')

        # Third groupby - use branch selection to get single sink
        mapped3 = cb.Graph(g3).map(
            pd.DataFrame({'e': [100, 200], 'p2': [0, 1], 'p3': ['A', 'B']}).set_index(
                'p2'
            )
        )
        mapped3['e'] = grouped2['gb2']
        grouped3 = mapped3.groupby('p3').reduce('f', name='gb3')

        result = grouped3.to_networkx()

        # Final grouped nodes should exist
        assert idx('gb3', 'A', dims=('p3',)) in result.nodes
        assert idx('gb3', 'B', dims=('p3',)) in result.nodes

    def test_groupby_then_map_then_groupby(self) -> None:
        """Test map → groupby → map → groupby sequence."""
        g1 = nx.DiGraph()
        g1.add_edge('a', 'b')

        g2 = nx.DiGraph()
        g2.add_edge('c', 'd')

        # First: map and groupby
        grouped1 = (
            cb.Graph(g1)
            .map(pd.DataFrame({'a': [1, 2, 3], 'p1': ['x', 'x', 'y']}))
            .groupby('p1')
            .reduce('b', name='gb1')
        )

        # Then: map again and groupby again - use branch selection
        mapped2 = cb.Graph(g2).map(
            pd.DataFrame({'c': [10, 20], 'p1': ['x', 'y'], 'p2': [0, 0]}).set_index(
                'p1'
            )
        )
        mapped2['c'] = grouped1['gb1']
        grouped2 = mapped2.groupby('p2').reduce('d', name='gb2')

        result = grouped2.to_networkx()

        # Should have single group for p2
        assert idx('gb2', 0, dims=('p2',)) in result.nodes

    def test_regular_reduce_then_groupby(self) -> None:
        """Test that regular reduce followed by groupby works."""
        g = nx.DiGraph()
        g.add_edge('a', 'b')
        g.add_edge('b', 'c')

        df = pd.DataFrame({'a': [1, 2, 3, 4], 'param': ['x', 'x', 'y', 'y']})
        graph = cb.Graph(g).map(df)

        # Regular reduce first
        reduced = graph.reduce('b', name='b_reduced')

        # Then groupby
        grouped = reduced.groupby('param').reduce('c', name='c_grouped')
        result = grouped.to_networkx()

        # Regular reduce node should exist
        assert 'b_reduced' in result.nodes

        # Grouped nodes should exist
        assert idx('c_grouped', 'x', dims=('param',)) in result.nodes
        assert idx('c_grouped', 'y', dims=('param',)) in result.nodes


class TestGroupbyWith2DNodes:
    """Tests for groupby operations on 2-D nodes (nodes with two dimensions)."""

    def test_2d_node_regular_reduce_then_groupby_reduce(self) -> None:
        """Test 2-D node → regular reduce along one dim → groupby-reduce the other."""
        g = nx.DiGraph()
        g.add_edge('a', 'c')
        g.add_edge('b', 'c')

        # Create 2-D node by mapping twice
        graph = cb.Graph(g)
        mapped = graph.map({'a': [1, 2, 3]}).map({'b': [10, 20]})
        # Now 'c' has 2 dimensions: dim_0 (length 3) and dim_1 (length 2)

        # First: regular reduce along dim_1
        reduced = mapped.reduce('c', name='reduced_c', index='dim_1')

        # Now 'reduced_c' has only dim_0, and we also have a 'b' column with values
        # Add grouping parameter aligned with dim_0
        reduced = reduced.map(
            pd.DataFrame({'param': ['x', 'x', 'y']}).set_index(
                pd.RangeIndex(3, name='dim_0')
            )
        )

        # Second: groupby-reduce along the remaining dimension
        grouped = reduced.groupby('param').reduce('reduced_c', name='final')
        result = grouped.to_networkx()

        # Check that final grouped nodes exist
        assert idx('final', 'x', dims=('param',)) in result.nodes
        assert idx('final', 'y', dims=('param',)) in result.nodes

        # Check that reduced_c nodes exist (one per dim_0 value)
        assert idx('reduced_c', 0, dims=('dim_0',)) in result.nodes
        assert idx('reduced_c', 1, dims=('dim_0',)) in result.nodes
        assert idx('reduced_c', 2, dims=('dim_0',)) in result.nodes

        # Check edges: reduced_c nodes should connect to appropriate final groups
        assert result.has_edge(
            idx('reduced_c', 0, dims=('dim_0',)), idx('final', 'x', dims=('param',))
        )
        assert result.has_edge(
            idx('reduced_c', 1, dims=('dim_0',)), idx('final', 'x', dims=('param',))
        )
        assert result.has_edge(
            idx('reduced_c', 2, dims=('dim_0',)), idx('final', 'y', dims=('param',))
        )

    def test_2d_node_groupby_then_regular_reduce(self) -> None:
        """Test 2-D node → groupby along one dim → regular reduce the other."""
        g = nx.DiGraph()
        g.add_edge('a', 'c')
        g.add_edge('b', 'c')

        # Create 2-D node with grouping parameter
        df = pd.DataFrame(
            {
                'a': [1, 2, 3],
                'param': ['x', 'x', 'y'],  # Will group dim_0
            }
        )
        graph = cb.Graph(g).map(df).map({'b': [10, 20]})
        # Now 'c' has 2 dimensions: dim_0 (length 3) and dim_1 (length 2)

        # First: groupby along dim_0 (using param)
        grouped = graph.groupby('param').reduce('c', name='grouped_c')
        # Now 'grouped_c' has only param dimension, but we still have dim_1

        # To perform regular reduce, we need to work with the grouped result
        # The grouped graph should still have dim_1 in its structure
        # Let's reduce along dim_1 within each group
        final = grouped.reduce('grouped_c', name='final', index='dim_1')
        result = final.to_networkx()

        # Check that final nodes exist (one per group, no dim_1)
        assert idx('final', 'x', dims=('param',)) in result.nodes
        assert idx('final', 'y', dims=('param',)) in result.nodes

        # Check intermediate grouped_c nodes exist (one per group per dim_1)
        # Note: axis order is (dim_1, param) not (param, dim_1)
        assert idx('grouped_c', 0, 'x', dims=('dim_1', 'param')) in result.nodes
        assert idx('grouped_c', 1, 'x', dims=('dim_1', 'param')) in result.nodes
        assert idx('grouped_c', 0, 'y', dims=('dim_1', 'param')) in result.nodes
        assert idx('grouped_c', 1, 'y', dims=('dim_1', 'param')) in result.nodes

        # Check edges: grouped_c nodes should connect to final reduced nodes
        assert result.has_edge(
            idx('grouped_c', 0, 'x', dims=('dim_1', 'param')),
            idx('final', 'x', dims=('param',)),
        )
        assert result.has_edge(
            idx('grouped_c', 1, 'x', dims=('dim_1', 'param')),
            idx('final', 'x', dims=('param',)),
        )
        assert result.has_edge(
            idx('grouped_c', 0, 'y', dims=('dim_1', 'param')),
            idx('final', 'y', dims=('param',)),
        )
        assert result.has_edge(
            idx('grouped_c', 1, 'y', dims=('dim_1', 'param')),
            idx('final', 'y', dims=('param',)),
        )
