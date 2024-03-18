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

    def to_tuple(self) -> tuple[tuple[IndexName, IndexValue]]:
        return tuple(zip(self.axes, self.values))

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
    graph: nx.DiGraph, *, successors: set[Hashable], index: IndexValues
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
    return nx.relabel_nodes(graph, renamed_nodes, copy=True)


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


MappingToArrayLike = Any  # dict[str, Numpy|DataArray], DataFrame, etc.


class Graph:
    def __init__(self, graph: nx.DiGraph, *, value_attr: str = 'value'):
        self.graph = graph
        self.indices: dict[IndexName, Iterable[IndexValue]] = {}
        self._value_attr = value_attr
        self._node_values: dict[tuple[IndexName, ...], MappingToArrayLike] = {}

    @property
    def value_attr(self) -> str:
        return self._value_attr

    @property
    def index_names(self) -> tuple[IndexName]:
        return tuple(self.indices)

    def map(self, node_values: MappingToArrayLike) -> Graph:
        """
        Map the graph over the given values by associating source nodes with values.

        All successors of the mapped source nodes are replaced with new nodes, one for
        each index value. The value is set as an attribute on the new source nodes
        (but not their successors).
        """
        for value_mapping in self._node_values.values():
            if any(node in value_mapping for node in node_values):
                raise ValueError('Node already has a value')
        root_nodes = tuple(node_values.keys())
        ndim = len(self.indices)
        indices = {}
        for name, values in _get_indices(node_values):
            if name is None:
                name = f'dim_{ndim}'
                ndim += 1
            indices[name] = values
        named = tuple(indices)
        if any([name in self.index_names for name in named]):
            raise ValueError(
                f'Conflicting new index names {named} with existing {self.index_names}'
            )
        successors = find_successors(self.graph, root_nodes=root_nodes)
        graph = self.graph.copy()
        for node in successors:
            graph.nodes[node]['indices'] = named + graph.nodes[node].get('indices', ())
        out = Graph(graph)
        # TODO order?
        out.indices = {**indices, **self.indices}
        out._node_values = dict(self._node_values)
        out._node_values[named] = node_values
        return out

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
        if key not in self.graph:
            raise KeyError(f"Node '{key}' does not exist in the graph.")
        indices: tuple[IndexName] = self.graph.nodes[key].get('indices', ())
        if index is not None and index not in indices:
            raise ValueError(f"Node '{key}' does not have index '{index}'.")
        # TODO We can support indexing from the back in the future.
        if axis is not None and (axis < 0 or axis >= len(indices)):
            raise ValueError(f"Node '{key}' does not have axis '{axis}'.")
        if index is not None:
            new_index = tuple(value for value in indices if value != index)
        elif axis is not None:
            # TODO Should axis refer to axes of graph, or the node?
            new_index = tuple(value for i, value in enumerate(indices) if i != axis)
        else:
            new_index = None
        indices_attr = {} if new_index is None else {'indices': new_index}
        if name in self.graph:
            raise ValueError(f'Node {name} already exists in the graph.')

        graph = self.graph.copy()
        graph.add_node(name, **attrs, **indices_attr)
        graph.add_edge(key, name)

        out = Graph(graph)
        out.indices = self.indices
        out._node_values = self._node_values
        return out

    def by_position(self, index_name: IndexName) -> PositionalIndexer:
        return PositionalIndexer(self, index_name)

    def to_networkx(self) -> nx.DiGraph:
        graph = self.graph
        for index_name, index in reversed(self.indices.items()):
            # Find all nodes with this index
            nodes = []
            for node, data in graph.nodes(data=True):
                if (node_indices := data.get('indices', None)) is not None:
                    if index_name in node_indices:
                        nodes.append(node)
            # Make a copy for each index value
            graphs = [
                rename_successors(
                    graph, successors=nodes, index=IndexValues.from_tuple(index)
                )
                for index in _yield_index([(index_name, index)])
            ]
            graph = nx.compose_all(graphs)

        # Get values using previously stored index values
        for values in self._node_values.values():
            for name, col in values.items():
                for match in [
                    node
                    for node in graph.nodes
                    if isinstance(node, NodeName) and node.name == name
                ]:
                    value = _get_value_at_index(col, match.index.to_tuple())
                    graph.nodes[match][self.value_attr] = value

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

    def __setitem__(self, key: int | slice, value: nx.Digraph) -> None:
        """
        Set the value of the index.

        Keeps the node and its attributes, but replaces the children with the children
        of the given value. The value must have a unique sink node. In other words,
        the subtree at key is replaced with the subtree at the sink node of value.
        """
        # TODO Do we actually want to do this conversion, or implement merging in a
        # way compatible with dims/indices of the graphs?
        if isinstance(value, Graph):
            value = value.to_networkx()
        sink_nodes = [node for node in value.nodes if value.out_degree(node) == 0]
        if len(sink_nodes) != 1:
            raise ValueError('Value must have exactly one sink node')
        sink = sink_nodes[0]
        sink_data = value.nodes[sink]
        graph = self.graph.copy()
        # ancestors = nx.ancestors(graph, key)
        # graph.remove_nodes_from(ancestors)
        self._remove_ancestors(graph, key)
        # graph.remove_edges_from(list(graph.in_edges(key)))
        graph.add_node(key, **sink_data)
        ancestor_graph = value.copy()
        ancestor_graph.remove_node(sink)

        # TODO Checks seem complicated, maybe we should just make it the user's
        # responsibility to ensure the graphs are compatible?
        # self._check_for_conflicts(graph, ancestor_graph)

        graph = nx.compose(graph, ancestor_graph)
        for parent in value.predecessors(sink):
            edge_data = value.get_edge_data(parent, sink)
            graph.add_edge(parent, key, **edge_data)
        # Delay setting graph until we know no step fails
        self.graph = graph

    def _remove_ancestors(self, graph: nx.DiGraph, node: Hashable) -> None:
        ancestors = nx.ancestors(graph, node)
        ancestors_successors = {
            ancestor: graph.successors(ancestor) for ancestor in ancestors
        }
        to_remove = []
        for ancestor, successors in ancestors_successors.items():
            # If any successor does not have node as descendant we must keep the node
            if all(nx.has_path(graph, successor, node) for successor in successors):
                to_remove.append(ancestor)
        graph.remove_nodes_from(to_remove)

    def _check_for_conflicts(self, graph: nx.DiGraph, ancestor_graph):
        for node in ancestor_graph.nodes:
            if node in graph:
                if graph.nodes[node] != ancestor_graph.nodes[node]:
                    raise ValueError(
                        f"Node '{node}' has different attributes in ancestor_graph"
                    )
                if list(graph.in_edges(node)) != list(ancestor_graph.in_edges(node)):
                    raise ValueError(
                        f"Node '{node}' has different incoming edges in ancestor_graph"
                    )
                # TODO The composite graph may add more edges, so this check is bad
                if list(graph.out_edges(node)) != list(ancestor_graph.out_edges(node)):
                    raise ValueError(
                        f"Node '{node}' has different outgoing edges in ancestor_graph"
                    )
