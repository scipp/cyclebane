# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx

IndexName = str | None
IndexValue = Hashable


@dataclass(frozen=True)
class IndexValues:
    axes: tuple[IndexName]
    values: tuple[IndexValue]

    @staticmethod
    # TODO fix type hint
    def from_tuple(t: tuple[IndexName, IndexValue]) -> IndexValues:
        return IndexValues(axes=t[::2], values=t[1::2])

    def merge_index(self, other: IndexValues) -> IndexValues:
        return IndexValues(
            axes=self.axes + other.axes, values=self.values + other.values
        )

    def pop(self, name: IndexName) -> IndexValues:
        i = self.axes.index(name)
        return IndexValues(
            axes=self.axes[:i] + self.axes[i + 1 :],
            values=self.values[:i] + self.values[i + 1 :],
        )

    def pop_axis(self, axis: int) -> IndexValues:
        if axis < 0 or axis >= len(self.axes):
            raise ValueError('Invalid axis')
        return IndexValues(
            axes=self.axes[:axis] + self.axes[axis + 1 :],
            values=self.values[:axis] + self.values[axis + 1 :],
        )

    def __str__(self):
        return ', '.join(
            f'{name}={value}' for name, value in zip(self.axes, self.values)
        )

    def __len__(self):
        return len(self.axes)


@dataclass(frozen=True)
class NodeName:
    name: str
    index: IndexValues

    def merge_index(self, other: IndexValues) -> NodeName:
        return NodeName(name=self.name, index=self.index.merge_index(other))

    def __str__(self):
        return f'{self.name}({self.index})'


def find_successors(graph: nx.DiGraph, *, root_nodes: tuple[Hashable]) -> set[Hashable]:
    successors = set()
    for root in root_nodes:
        if root not in graph:
            raise ValueError("Node not in graph")
        if graph.in_degree(root) > 0:
            raise ValueError("Node is not a root node")
        nodes = nx.dfs_successors(graph, root)
        successors.update(
            set(node for node_list in nodes.values() for node in node_list)
        )
        successors.add(root)
    return successors


def rename_successors(
    graph: nx.DiGraph,
    *,
    successors: set[Hashable],
    index: IndexValues,
    values: dict[Hashable, Any],
    value_attr: str = 'value',
) -> nx.DiGraph:
    """Replace 'node' and all its successors with (node, suffix), and update all edges
    accordingly."""
    renamed_nodes = {
        node: (
            node.merge_index(index)
            if isinstance(node, NodeName)
            else NodeName(name=node, index=index)
        )
        for node in successors
    }
    relabeled = nx.relabel_nodes(graph, renamed_nodes, copy=True)
    for root, value in values.items():
        relabeled.nodes[renamed_nodes[root]][value_attr] = value
    return relabeled


def add_node_attr(
    graph: nx.DiGraph, node: Hashable, attr: dict[str, Hashable]
) -> nx.DiGraph:
    """Add attributes to a node."""
    graph = graph.copy()
    graph.nodes[node].update(attr)
    return graph


@dataclass
class Index:
    name: str
    values: Iterable[Hashable]


class Graph:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.labels: dict[Hashable, list[Hashable]] = {}

    def get_index_names(self, values) -> tuple[str]:
        if (dims := getattr(values, 'dims', None)) is not None:
            return dims
        if (ndim := getattr(values, 'ndim', None)) is not None:
            if (index := getattr(values, 'index', None)) is not None:
                # TODO There can be multiple names in Pandas?
                if index.name is not None:
                    return index.name
        else:
            ndim = 1  # TODO nested lists?
        return (None,) * ndim

    def get_shape(self, values) -> tuple[int]:
        if (shape := getattr(values, 'shape', None)) is not None:
            return shape
        return (len(values),)

    def yield_index(self, index_names: tuple[str], shape: tuple[int]):
        """Given a multi-dimensional index, yield all possible combinations."""
        if len(index_names) != len(shape):
            raise ValueError('Length of index_names and shape must be the same')
        for i in range(shape[0]):
            if len(index_names) == 1:
                yield (index_names[0], i)
            else:
                for rest in self.yield_index(index_names[1:], shape[1:]):
                    yield (index_names[0], i) + rest

    def _get_value_at_index(self, values: Iterable[Any], index: tuple[str, int]) -> Any:
        for label, i in zip(index[::2], index[1::2]):
            if label is None:
                values = values[i]
            else:
                # TODO this is Scipp notation? Support also Pandas and Xarray.
                values = values[(label, i)]
        return values

    def map(
        self,
        node_values: Mapping[Hashable, Iterable[Any]],
        value_attr: str = 'value',
    ) -> Graph:
        """For every value, create a new graph with all successors renamed, merge all
        resulting graphs."""
        root_nodes = tuple(node_values.keys())
        shapes = [self.get_shape(values) for values in node_values.values()]
        if len(set(shapes)) != 1:
            raise ValueError('All values must have the same shape')
        index_names = self.get_index_names(next(iter(node_values.values())))
        shape = shapes[0]
        successors = find_successors(self.graph, root_nodes=root_nodes)
        graphs = [
            rename_successors(
                self.graph,
                successors=successors,
                index=IndexValues.from_tuple(index),
                values={
                    root: self._get_value_at_index(vals, index)
                    for root, vals in node_values.items()
                },
                value_attr=value_attr,
            )
            for index in self.yield_index(index_names, shape)
        ]
        graph = Graph(nx.compose_all(graphs))
        # graph.labels = {**self.labels}
        # graph.labels[index_name] = (index_name, values)
        return graph

    def __getitem__(self, sel: tuple[str, int]) -> Graph:
        """
        Return a new graph, essentially undoing the effect of `map`.

        Remove any node that has (key, i) for i != index, and remove the key from the
        labels.
        """
        key, index = sel
        graph = self.graph.copy()
        drop = []
        remain = []
        for node in self.graph.nodes:
            if isinstance(node, tuple):
                name, *indices = node
                if name == key and isinstance(indices[0], int):
                    if indices[0] == index:
                        remain.append(node)
                    else:
                        drop.append(node)
                elif name != key:
                    for dim, i in indices:
                        if dim == key:
                            if i == index:
                                remain.append(node)
                            else:
                                drop.append(node)
                            break
        for node in drop:
            if node[1] != index:
                graph.remove_node(node)
        # TODO remove index from remaining nodes, replace root by label
        print(remain)
        graph = Graph(graph)
        graph.labels = {**self.labels}
        del graph.labels[key]
        return graph

    def reduce(
        self,
        key: str,
        *,
        index: None | Hashable = None,
        axis: None | int = None,
        func: str,
    ) -> Graph:
        """Add edges from all nodes (key, index) to new node func."""
        # TODO Should not use func as nodename. Func should be metadata, i.e.,
        # arbitrary attrs to store in new nodes
        if index is not None and axis is not None:
            raise ValueError('Only one of index and axis can be given')
        nodes = [
            node
            for node in self.graph.nodes
            if isinstance(node, NodeName) and node.name == key
        ]
        graph = self.graph.copy()
        for node in nodes:
            if index is not None:
                new_index = node.index.pop(index)
            elif axis is not None:
                new_index = node.index.pop_axis(axis)
            new_node = (
                func if len(new_index) == 0 else NodeName(name=func, index=new_index)
            )
            graph.add_edge(node, new_node)
        graph = Graph(graph)
        # for name, labels in self.labels.items():
        #    if labels[0] != index:
        #        graph.labels[name] = labels
        return graph

    def groupby(self, key: str, index: str, reduce: str) -> Graph:
        """
        Similar to reduce, but group nodes by label.

        Add edges from all nodes (key, index) to new node (func, label), for every
        label.  The label is given by `labels[index]`.
        """
        nodes = [node for node in self.graph.nodes if node[0] == key]
        orig_index, labels = self.labels[index]
        sorted_unique_labels = sorted(set(labels))
        graph = self.graph.copy()
        for node in nodes:
            # Node looks like (key, (orig_index, 2), ('y', 11))
            # We want to add an edge to (reduce, (index, label), ('y', 11))
            _, *indices = node
            orig_pos = [i[1] for i in indices if i[0] == orig_index][0]
            orig_label = labels[orig_pos]
            indices = [i for i in indices if i[0] != orig_index]
            label = sorted_unique_labels.index(orig_label)
            graph.add_edge(node, (reduce, (index, label), *indices))
        graph = Graph(graph)
        for name, labels in self.labels.items():
            if labels[0] != orig_index:
                graph.labels[name] = labels
        graph.labels[index] = (index, sorted_unique_labels)
        return graph

    def _repr_html_(self):
        from IPython.display import display
        from networkx.drawing.nx_agraph import to_agraph

        A = to_agraph(self.graph)
        A.layout('dot')
        display(A)
