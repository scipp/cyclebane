# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Tests ensuring that graphs are reclaimable by reference counting.

Graphs (and thus the data stored on their nodes) that are part of a reference cycle
are freed only by the cyclic garbage collector. That collector is triggered by the
number of allocated objects, so few but large objects may survive indefinitely.
"""

import gc
import weakref
from collections.abc import Iterator
from contextlib import contextmanager

import networkx as nx
import pytest

import cyclebane as cb


@contextmanager
def refcount_only_gc() -> Iterator[None]:
    """Disable the cyclic garbage collector, leaving only reference counting."""
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


def live_graphs() -> list[nx.DiGraph]:
    return [obj for obj in gc.get_objects() if isinstance(obj, nx.DiGraph)]


class Payload:
    """Stand-in for the (potentially large) data stored on a node."""


def set_value(graph: cb.Graph, node: str, value: Payload) -> None:
    """Set a node value the way, e.g., Sciline inserts a parameter into a pipeline."""
    branch = nx.DiGraph()
    branch.add_node(node, value=value)
    graph[node] = cb.Graph(branch)


@pytest.fixture
def graph() -> cb.Graph:
    return cb.Graph(nx.DiGraph([('a', 'b'), ('b', 'c')]))


def test_graph_does_not_form_cycle_when_taking_views() -> None:
    with refcount_only_gc():
        graph = cb.Graph(nx.DiGraph([('a', 'b'), ('b', 'c')]))
        ref = weakref.ref(graph.graph)
        list(graph.graph.edges)
        list(graph.graph.in_edges('b'))
        assert graph.graph.out_degree('c') == 0
        del graph
        assert ref() is None


def test_setitem_leaves_no_garbage(graph: cb.Graph) -> None:
    branch = cb.Graph(nx.DiGraph([('x', 'b')]))
    with refcount_only_gc():
        before = {id(g) for g in live_graphs()}
        graph['b'] = branch
        garbage = [
            g for g in live_graphs() if id(g) not in before and g is not graph.graph
        ]
        assert garbage == []


def test_setitem_frees_value_of_replaced_node(graph: cb.Graph) -> None:
    with refcount_only_gc():
        set_value(graph, 'a', Payload())
        ref = weakref.ref(graph.graph.nodes['a']['value'])
        set_value(graph, 'a', Payload())
        assert ref() is None


def test_delitem_leaves_graph_reclaimable(graph: cb.Graph) -> None:
    with refcount_only_gc():
        del graph['c']
        ref = weakref.ref(graph.graph)
        graph.graph = None
        assert ref() is None


def test_to_networkx_result_is_freed_after_inspecting_it(graph: cb.Graph) -> None:
    mapped = graph.map({'a': [1, 2, 3]}).reduce('c', name='sum')
    with refcount_only_gc():
        result = mapped.to_networkx()
        # Consumers of the returned graph inspect it, which materializes views.
        assert len(list(result.edges)) > 0
        assert result.in_degree('sum') == 3
        ref = weakref.ref(result)
        del result
        assert ref() is None


def test_map_reduce_to_networkx_leaves_no_garbage(graph: cb.Graph) -> None:
    graph.map({'a': [1, 2, 3]}).reduce('c', name='sum').to_networkx()
    with refcount_only_gc():
        before = {id(g) for g in live_graphs()}
        graph.map({'a': [1, 2, 3]}).reduce('c', name='sum').to_networkx()
        garbage = [g for g in live_graphs() if id(g) not in before]
        assert garbage == []
