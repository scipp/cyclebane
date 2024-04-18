# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Hashable, Iterable, Mapping, Sequence
from uuid import uuid4

import networkx as nx

IndexName = Hashable
IndexValue = Hashable


def _is_pandas_series_or_dataframe(obj: Any) -> bool:
    return str(type(obj)) in [
        "<class 'pandas.core.frame.DataFrame'>",
        "<class 'pandas.core.series.Series'>",
    ]


def _get_unique_sink(graph: nx.DiGraph) -> Hashable:
    sink_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
    if len(sink_nodes) != 1:
        raise ValueError('Graph must have exactly one sink node')
    return sink_nodes[0]


def _get_new_node_name(graph: nx.DiGraph) -> str:
    while True:
        name = str(uuid4())
        if name not in graph:
            return name


def _remove_ancestors(graph: nx.DiGraph, node: Hashable) -> nx.DiGraph:
    graph = graph.copy()
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
    graph.remove_edges_from(list(graph.in_edges(node)))
    return graph


def _check_for_conflicts(graph: nx.DiGraph, ancestor_graph):
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


@dataclass(frozen=True)
class MappedNode:
    name: Hashable
    indices: tuple[int, ...]


def node_with_indices(node: Hashable, indices: tuple[int, ...]) -> MappedNode:
    if isinstance(node, MappedNode):
        return MappedNode(name=node.name, indices=indices + node.indices)
    return MappedNode(name=node, indices=indices)


def node_indices(node: Hashable) -> tuple[int, ...] | None:
    if isinstance(node, MappedNode):
        return node.indices
    return ()


def _find_successors(
    graph: nx.DiGraph, *, root_nodes: tuple[Hashable]
) -> set[Hashable]:
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


def _rename_successors(
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

        def slice_values(
            index_names: tuple[IndexName, ...], values: MappingToArrayLike
        ) -> MappingToArrayLike:
            if self.index_name not in index_names:
                return values
            return {
                name: col[start:stop:step] if name == self.index_name else col
                for name, col in values.items()
            }

        out.indices = {
            name: slice_index(name, index) for name, index in self.graph.indices.items()
        }
        out._node_values = {
            index_names: slice_values(index_names, values)
            for index_names, values in self.graph._node_values.items()
        }
        return out


MappingToArrayLike = Any  # dict[str, Numpy|DataArray], DataFrame, etc.


class Graph:
    """
    A Cyclebane graph is a directed acyclic graph with additional array-like structure.

    The array-like structure selectively affects nodes in the graph by associating
    source nodes with an array-like object. The source node and all its descendants
    thus gain an additional index or dimension.

    Notes
    -----
    The current implementation is a proof of concept, there is a number of things to
    improve:
    - I think I want to avoid spelling out the indices early in `map`, but instead delay
      this until `to_networkx`.
    - Overall, I would like to reduce the array-handling code and transparently forward
      to the slicing code of the underlying array-like object (Pandas, NumPy, Xarray,
      Scipp). Basically, we would like to use the slicing methods of the underlying
      object. This may not be trivial, since we might mix different types of array-like
      objects at nodes with multiple predecessors.
    - We could avoid the `indices` attribute on nodes (added by `map`). Instead, lookup
      the ancestors and identify mapped source nodes. This will simplify slicing, as
      it avoids the need to remove indices on nodes (we only slice the values arrays).
      `reduce` would need to add an attribute on which dim to reduce though, so this
      may not actually be easier. It might solve the problem of selecting branches
      though, which also needs to select a subset of the mapped arrays.
    """

    def __init__(self, graph: nx.DiGraph, *, value_attr: str = 'value'):
        self.graph = graph
        self.indices: dict[IndexName, Iterable[IndexValue]] = {}
        self._value_attr = value_attr
        self._node_values: dict[tuple[IndexName, ...], MappingToArrayLike] = {}

    def copy(self) -> Graph:
        graph = Graph(self.graph.copy(), value_attr=self._value_attr)
        graph.indices = dict(self.indices)
        graph._node_values = dict(self._node_values)
        return graph

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
        successors = _find_successors(self.graph, root_nodes=root_nodes)
        name_mapping: dict[Hashable, MappedNode] = {}
        for node in successors:
            name_mapping[node] = node_with_indices(node, named)
        graph = nx.relabel_nodes(self.graph, name_mapping)

        out = Graph(graph)
        # TODO order?
        out.indices = {**indices, **self.indices}
        out._node_values = dict(self._node_values)
        out._node_values[named] = node_values
        return out

    def reduce(
        self,
        key: None | str = None,
        *,
        index: None | Hashable = None,
        axis: None | int = None,
        name: None | str = None,
        attrs: None | dict[str, Any] = None,
    ) -> Graph:
        """
        Reduce over the given index or axis previously created with :py:meth:`map`.

        If neither index nor axis is given, all axes are reduced.

        Parameters
        ----------
        key:
            The name of the source node to reduce. This is the original name prior to
            mapping. If not given, tries to find a unique sink node.
        index:
            The name of the index to reduce over. Only one of index and axis can be
            given.
        axis:
            The axis to reduce over. Only one of index and axis can be given.
        name:
            The name of the new node. If not given, a unique name is generated.
        attrs:
            Attributes to set on the new node(s).
        """
        key = key or _get_unique_sink(self.graph)
        name = name or _get_new_node_name(self.graph)

        attrs = attrs or {}
        if index is not None and axis is not None:
            raise ValueError('Only one of index and axis can be given')
        key = self._from_orig_key(key)
        indices: tuple[IndexName] = node_indices(key)
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
        if name in self.graph:
            raise ValueError(f'Node {name} already exists in the graph.')

        graph = self.graph.copy()
        name = MappedNode(name=name, indices=new_index) if new_index else name
        graph.add_node(name, **attrs)
        graph.add_edge(key, name)

        out = Graph(graph)
        out.indices = dict(self.indices)
        out._node_values = dict(self._node_values)
        return out

    def _from_orig_key(self, key: Hashable) -> Hashable:
        # Graph.map relabels nodes to include index names, which can be inconvenient
        # for the user. Is this convenience of finding the node by its original name
        # worth the complexity and a good idea?
        if key not in self.graph:
            matches = [
                node
                for node in self.graph.nodes
                if isinstance(node, MappedNode) and node.name == key
            ]
            if len(matches) == 0:
                raise KeyError(f"Node '{key}' does not exist in the graph.")
            if len(matches) > 1:
                raise KeyError(f"Node '{key}' is ambiguous. Found {matches}.")
            return matches[0]
        return key

    def by_position(self, index_name: IndexName) -> PositionalIndexer:
        return PositionalIndexer(self, index_name)

    def to_networkx(self) -> nx.DiGraph:
        graph = self.graph
        for index_name, index in reversed(self.indices.items()):
            # Find all nodes with this index
            nodes = []
            for node in graph.nodes():
                if index_name in node_indices(
                    node.name if isinstance(node, NodeName) else node
                ):
                    nodes.append(node)
            # Make a copy for each index value
            graphs = [
                _rename_successors(
                    graph, successors=nodes, index=IndexValues.from_tuple(index)
                )
                for index in _yield_index([(index_name, index)])
            ]
            graph = nx.compose_all(graphs)
        # Replace all MappingNodes with their name
        new_names = {
            node: NodeName(node.name.name, node.index)
            for node in graph
            if isinstance(node, NodeName)
        }
        graph = nx.relabel_nodes(graph, new_names)

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

    def __getitem__(self, key: Hashable | slice) -> Graph:
        """
        Get the branch of the graph rooted at the given node.

        The branch is a subgraph containing the given node and all its ancestors.
        Think of this like a Git branch, where the given node is the head of the branch.
        """
        if isinstance(key, slice):
            raise NotImplementedError('Only single nodes are supported ')
        key = self._from_orig_key(key)
        ancestors = nx.ancestors(self.graph, key)
        ancestors.add(key)
        out = Graph(self.graph.subgraph(ancestors))
        # TODO Only keep indices and values for nodes in the subgraph
        out.indices = dict(self.indices)
        out._node_values = dict(self._node_values)
        return out

    def __setitem__(self, branch: Hashable | slice, other: Graph) -> None:
        """
        Set a new branch in place of the given branch.

        The new branch must have a unique sink node. The branch at `branch` is replaced
        with the new branch. The indices and node values are updated accordingly. The
        edges to successors of the old branch are connected to the sink of the new
        branch.
        """
        if not isinstance(other, Graph):
            raise TypeError(f'Expected {Graph}, got {type(other)}')
        new_branch = other.graph
        sink = _get_unique_sink(new_branch)
        new_branch = nx.relabel_nodes(new_branch, {sink: branch})
        graph = _remove_ancestors(self.graph, branch)
        graph.nodes[branch].clear()

        # TODO Checks seem complicated, maybe we should just make it the user's
        # responsibility to ensure the graphs are compatible?
        # _check_for_conflicts(graph, ancestor_graph)

        graph = nx.compose(graph, new_branch)

        # Delay setting graph until we know no step fails
        self.graph = graph
        self.indices.update(other.indices)
        self._node_values.update(other._node_values)
