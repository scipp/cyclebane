# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Hashable, Iterable, Mapping, Sequence

import networkx as nx

IndexName = Hashable
IndexValue = Hashable


def _is_pandas_series_or_dataframe(obj: Any) -> bool:
    return str(type(obj)) in [
        "<class 'pandas.core.frame.DataFrame'>",
        "<class 'pandas.core.series.Series'>",
    ]


@dataclass(frozen=True)
class IndexValues:
    axes: tuple[IndexName]
    values: tuple[IndexValue]

    @staticmethod
    def from_tuple(t: tuple[tuple[IndexName, IndexValue]]) -> IndexValues:
        names = tuple(name for name, _ in t)
        values = tuple(value for _, value in t)
        return IndexValues(axes=names, values=values)

    def merge_index(self, other: IndexValues) -> IndexValues:
        return IndexValues(
            axes=other.axes + self.axes, values=other.values + self.values
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


def _get_indices(
    node_values: Mapping[Hashable, Sequence[Any]]
) -> list[tuple[IndexName, Iterable[IndexValue]]]:
    col_values = _get_col_values(node_values)
    # We do not descend into nested lists, users should use, e.g., NumPy if
    # they want to do that.
    shapes = [getattr(col, 'shape', (len(col),)) for col in col_values]
    if len(set(shapes)) != 1:
        raise ValueError(
            'All value sequences in a map operation must have the same shape. '
            'Use multiple map operations if necessary.'
        )
    shape = shapes[0]

    # TODO Catch cases where different items have different indices?
    values = next(iter(col_values))
    if (dims := getattr(values, 'dims', None)) is not None:
        # Note that we are currently not attempting to use Xarray or Scipp coords
        # as indices.
        sizes = dict(zip(dims, values.shape))
        return [(dim, range(sizes[dim])) for dim in dims]
    if _is_pandas_series_or_dataframe(values):
        # TODO There can be multiple names in Pandas?
        return [(values.index.name, values.index)]
    else:
        return [(None, range(size)) for size in shape]


def _get_col_values(values: Mapping[Hashable, Sequence[Any]]) -> list[Sequence[Any]]:
    if (columns := getattr(values, 'columns', None)) is not None:
        return [values.iloc[:, i] for i in range(len(columns))]
    return list(values.values())


def _yield_index(
    indices: list[tuple[IndexName, Iterable[IndexValue]]]
) -> Generator[tuple[tuple[IndexName, IndexValue], ...], None, None]:
    """Given a multi-dimensional index, yield all possible combinations."""
    name, index = indices[0]
    for index_value in index:
        if len(indices) == 1:
            yield ((name, index_value),)
        else:
            for rest in _yield_index(indices[1:]):
                yield ((name, index_value),) + rest


def _get_value_at_index(
    values: Sequence[Any], index_values: list[tuple[IndexName, IndexValue]]
) -> Any:
    if hasattr(values, 'isel'):
        return values.isel(dict(index_values))
    for label, i in index_values:
        # TODO Fix the condition for automatic label detection. We can we ensure we
        # index the correct axis? Should we just use an integer axis index?
        if (
            label is None
            or label.startswith('dim_')
            or (hasattr(values, 'ndim') and values.ndim == 1)
        ):
            values = values[i]
        else:
            # This is Scipp notation, Xarray uses the 'isel' method.
            values = values[(label, i)]
    return values


class PositionalIndexer:
    def __init__(self, graph: Graph, index_name: IndexName):
        self.graph = graph
        self.index_name = index_name

    def __getitem__(self, key: int | slice) -> Graph:
        if isinstance(key, int):
            raise ValueError('Only slices are supported')
        start, stop, step = key.start, key.stop, key.step
        out = Graph(self.graph.graph)

        def slice_index(
            name: IndexName, index: Iterable[IndexValue]
        ) -> Iterable[IndexValue]:
            if name != self.index_name:
                return index
            return index[start:stop:step]

        out.indices = {
            name: slice_index(name, index) for name, index in self.graph.indices.items()
        }
        return out


class Graph:
    def __init__(self, graph: nx.DiGraph, *, value_attr: str = 'value'):
        self.graph = graph
        self.indices: dict[IndexName, Iterable[IndexValue]] = {}
        self._value_attr = value_attr
        self._node_values: dict[Hashable, Sequence[Any]] = {}

    @property
    def value_attr(self) -> str:
        return self._value_attr

    @property
    def index_names(self) -> tuple[IndexName]:
        return tuple(self.indices)

    def map(self, node_values: Mapping[Hashable, Sequence[Any]]) -> Graph:
        """
        Map the graph over the given values by associating source nodes with values.

        All successors of the mapped source nodes are replaced with new nodes, one for
        each index value. The value is set as an attribute on the new source nodes
        (but not their successors).
        """
        if any(node in self._node_values for node in node_values):
            raise ValueError('Node already has a value')
        root_nodes = tuple(node_values.keys())
        ndim = len(self.indices)
        indices = {}
        for name, values in _get_indices(node_values):
            if name is None:
                ndim += 1
                name = f'dim_{ndim}'
            indices[name] = values
        named = tuple(indices)
        if any([name in self.index_names for name in named]):
            raise ValueError(
                f'Conflicting new index names {named} with existing {self.index_names}'
            )
        successors = find_successors(self.graph, root_nodes=root_nodes)
        graph = self.graph.copy()
        for node in successors:
            graph.nodes[node]['indices'] = named
        out = Graph(graph)
        # TODO order?
        out.indices = {**self.indices, **indices}
        print(out.indices)
        out._node_values = {
            **self._node_values,
            **dict(zip(root_nodes, _get_col_values(node_values))),
        }
        return out

    def by_position(self, index_name: IndexName) -> PositionalIndexer:
        return PositionalIndexer(self, index_name)

    def to_networkx(self) -> nx.DiGraph:
        graph = self.graph
        for index_name, index in self.indices.items():
            print(index_name, index)
            # Find all nodes with this index
            nodes = []
            for node, data in graph.nodes(data=True):
                if (node_indices := data.get('indices', None)) is not None:
                    if index_name in node_indices:
                        nodes.append(node)
            # Make a copy for each index value
            graphs = [
                rename_successors(
                    graph,
                    successors=nodes,
                    index=IndexValues.from_tuple(index),
                    values={
                        # TODO This is not correct if values are > 1D
                        root: _get_value_at_index(vals, index)
                        for root, vals in self._node_values.items()
                        if root in nodes
                    },
                    value_attr=self.value_attr,
                )
                for index in _yield_index([(index_name, index)])
            ]
            graph = nx.compose_all(graphs)
        return graph

    def __getitem__(self, key: str | slice) -> Any:
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if stop is not None or step is not None:
                raise ValueError('Only start is supported')
            ancestors = nx.ancestors(self.graph, start)
            ancestors.add(start)
            out = Graph(self.graph.subgraph(ancestors))
            out.indices = self.indices
            return out

        # TODO not quite correct if we have mapping
        return self.graph.nodes[key]


class GraphOld:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    @property
    def index_names(self) -> set[IndexName]:
        return self.graph.graph.get('index_names', set())

    def map(
        self,
        node_values: Mapping[Hashable, Sequence[Any]],
        *,
        value_attr: str = 'value',
    ) -> Graph:
        """
        Map the graph over the given values by associating source nodes with values.

        All successors of the mapped source nodes are replaced with new nodes, one for
        each index value. The value is set as an attribute on the new source nodes
        (but not their successors).
        """
        root_nodes = tuple(node_values.keys())
        indices = _get_indices(node_values)
        named = tuple(name for name, _ in indices if name is not None)
        if any([name in self.index_names for name in named]):
            raise ValueError(
                f'Conflicting new index names {named} with existing {self.index_names}'
            )
        successors = find_successors(self.graph, root_nodes=root_nodes)
        graphs = [
            rename_successors(
                self.graph,
                successors=successors,
                index=IndexValues.from_tuple(index),
                values={
                    root: _get_value_at_index(vals, index)
                    for root, vals in node_values.items()
                },
                value_attr=value_attr,
            )
            for index in _yield_index(indices=indices)
        ]
        graph = nx.compose_all(graphs)
        graph.graph['index_names'] = self.index_names | set(named)
        return Graph(graph)

    def reduce(
        self,
        key: str,
        *,
        index: None | Hashable = None,
        axis: None | int = None,
        name: str,
        attrs: None | dict[str, Any] = None,
    ) -> Graph:
        """
        Reduce over the given index or axis previously created with :py:meth:`map`.
        `

        If neither index nor axis is given, all axes are reduced.

        Parameters
        ----------
        key:
            The name of the source node to reduce. This is the original name prior to
            mapping. Note that there is ambiguity if the same was used as 'name' in
            a previous reduce operation over a subset of indices/axes.
        index:
            The name of the index to reduce over. Only one of index and axis can be
            given.
        axis:
            The axis to reduce over. Only one of index and axis can be given.
        name:
            The name of the new node(s). If not all axes of the node identified by
            the key are reduced then this will be the name property of the
            :py:class:`NodeName` instances used to identify the new nodes.
        attrs:
            Attributes to set on the new node(s).
        """
        attrs = attrs or {}
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
            else:
                new_index = IndexValues(axes=(), values=())
            new_node = (
                name if len(new_index) == 0 else NodeName(name=name, index=new_index)
            )
            # The elif checks self.graph, not graph, since previous iteration may have
            # inserted the node already. Note that we do need to handle multiple
            # inserts because in general not all axes are reduced, so there are
            # multiple new nodes.
            if new_node not in graph:
                graph.add_node(new_node, **attrs)
            elif new_node in self.graph:
                raise ValueError(
                    f'Node {new_node} already exists in the graph. '
                    'Use a different name or reduce over a subset of indices/axes.'
                )
            graph.add_edge(node, new_node)

        return Graph(graph)
