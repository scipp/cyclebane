# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Generator, Hashable, Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import networkx as nx

from .node_values import IndexName, IndexValue, NodeValues


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
    graph_without_node = graph.copy()
    graph_without_node.remove_node(node)
    ancestors = nx.ancestors(graph, node)
    # Considering the graph we obtain by removing `node`, we need to consider the
    # descendants of each ancestor. If an ancestor has descendants that are not
    # removal candidates, we should not remove the ancestor.
    to_remove = [
        ancestor
        for ancestor in ancestors
        if nx.descendants(graph_without_node, ancestor).issubset(ancestors)
    ]
    graph = graph.copy()
    graph.remove_nodes_from(to_remove)
    graph.remove_edges_from(list(graph.in_edges(node)))
    return graph


@dataclass(frozen=True, slots=True)
class IndexValues:
    """
    Index values used as part of :py:class:`NodeName`.

    Conceptually, this is a mapping from index names to index values.
    """

    axes: tuple[IndexName, ...]
    values: tuple[IndexValue, ...]

    @staticmethod
    def from_tuple(t: tuple[tuple[IndexName, IndexValue], ...]) -> IndexValues:
        names = tuple(name for name, _ in t)
        values = tuple(value for _, value in t)
        return IndexValues(axes=names, values=values)

    def to_tuple(self) -> tuple[tuple[IndexName, IndexValue], ...]:
        return tuple(zip(self.axes, self.values, strict=True))

    def merge_index(self, other: IndexValues) -> IndexValues:
        return IndexValues(
            axes=other.axes + self.axes, values=other.values + self.values
        )

    def __str__(self) -> str:
        return ', '.join(
            f'{name}={value}'
            for name, value in zip(self.axes, self.values, strict=True)
        )

    def __len__(self) -> int:
        return len(self.axes)


@dataclass(frozen=True, slots=True)
class NodeName:
    """Node name with indices used for mapped nodes when converting to NetworkX."""

    name: Hashable
    index: IndexValues

    def merge_index(self, other: IndexValues) -> NodeName:
        return NodeName(name=self.name, index=self.index.merge_index(other))

    def __str__(self) -> str:
        return f'{self.name}({self.index})'


@dataclass(frozen=True, slots=True)
class MappedNode:
    """
    Key for a node in :py:class:`Graph` representing a collection of "mapped" nodes.
    """

    name: Hashable
    indices: tuple[IndexName, ...]


def _node_with_indices(node: Hashable, indices: tuple[IndexName, ...]) -> MappedNode:
    if isinstance(node, MappedNode):
        return MappedNode(name=node.name, indices=indices + node.indices)
    return MappedNode(name=node, indices=indices)


def _node_indices(node: Hashable) -> tuple[IndexName, ...]:
    if isinstance(node, MappedNode):
        return node.indices
    return ()


def _find_successors(
    graph: nx.DiGraph, *, root_nodes: tuple[Hashable]
) -> set[Hashable]:
    successors = set()
    for root in root_nodes:
        if graph.in_degree(root) > 0:
            raise ValueError(f"Mapped node '{root}' is not a source node")
        successors.update(nx.descendants(graph, source=root) | {root})
    return successors


def _rename_successors(
    graph: nx.DiGraph, *, successors: Iterable[Hashable], index: IndexValues
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


def _yield_index(
    indices: list[tuple[IndexName, Iterable[IndexValue]]],
) -> Generator[tuple[tuple[IndexName, IndexValue], ...], None, None]:
    """Given a multi-dimensional index, yield all possible combinations."""
    name, index = indices[0]
    for index_value in index:
        if len(indices) == 1:
            yield ((name, index_value),)
        else:
            for rest in _yield_index(indices[1:]):
                yield ((name, index_value), *rest)


class PositionalIndexer:
    """
    Helper class to allow slicing a named dim of a graph using positional indexing.
    """

    def __init__(self, graph: Graph, index_name: IndexName):
        self.graph = graph
        self.index_name = index_name

    def __getitem__(self, key: int | slice) -> Graph:
        # Supporting single indices may be conceptually ill-defined if the index
        # `reduce` was applied to the graph, so we might never support this.
        if isinstance(key, int):
            raise NotImplementedError('Only slices are supported')
        node_values = NodeValues(
            {
                name: (
                    col.loc({self.index_name: key})
                    if self.index_name in col.index_names
                    else col
                )
                for name, col in self.graph._node_values.items()
            }
        )
        return Graph(self.graph.graph, node_values=node_values)


MappingToArrayLike = Any  # dict[str, Numpy|DataArray], DataFrame, etc.


class Graph:
    """
    A Cyclebane graph is a directed acyclic graph with additional array-like structure.

    The array-like structure selectively affects nodes in the graph by associating
    source nodes with an array-like object. These source node and all their descendants
    thus gain an additional index (or "dimension").

    Nomenclature:

    - Index: As in Pandas, and index is a sequence of values that label an axis.
    - Index-value: A single value in an index.
    - Index-name: The name of an index.


    Notes
    -----
    The current implementation is not complete, there is a number of things to
    improve:
    - Overall, I would like to reduce the array-handling code and transparently forward
      to the slicing code of the underlying array-like object (Pandas, NumPy, Xarray,
      Scipp). Basically, we would like to use the slicing methods of the underlying
      object. This may not be trivial, since we might mix different types of array-like
      objects at nodes with multiple predecessors.
    """

    def __init__(self, graph: nx.DiGraph, *, node_values: NodeValues | None = None):
        """
        Initialize a graph from a directed NetworkX graph.

        Parameters
        ----------
        graph:
            The directed graph representing the data flow.
        node_values:
            A mapping from source node names to array-like objects. The implementation
            assumes that the graph has been setup correctly. Do not use this argument
            unless you know what you are doing.
        """
        self.graph = graph
        self._node_values = node_values or NodeValues({})

    def copy(self) -> Graph:
        return Graph(self.graph.copy(), node_values=self._node_values)

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        """Names of the indices (dimensions) of the graph."""
        return tuple(self.indices)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        """Names and values of the indices of the graph."""
        return self._node_values.indices

    def map(self, node_values: MappingToArrayLike) -> Graph:
        """
        Map the graph over the given values by associating source nodes with values.

        All successors of the mapped source nodes are replaced with new nodes, one for
        each index value. The value is set as an attribute on the new source nodes
        (but not their successors).

        Parameters
        ----------
        node_values:
            A mapping from source node names to array-like objects. The source nodes
            are the roots of the branches to be mapped. The array-like objects must
            support slicing, e.g., NumPy arrays, Xarray DataArrays, Pandas DataFrames,
            etc.
        """
        new_values = NodeValues.from_mapping(
            node_values, axis_zero=len(self.index_names)
        )

        # Make sure root nodes exist in graph, add them if not. This choice allows for
        # mapping, e.g., with multiple columns from a DataFrame, representing labels
        # used later for groupby operations.
        graph = self.graph.copy()
        graph.add_nodes_from(new_values)

        successors = _find_successors(graph, root_nodes=new_values)
        name_mapping: dict[Hashable, MappedNode] = {}
        for node in successors:
            name_mapping[node] = _node_with_indices(node, tuple(new_values.indices))

        return Graph(
            nx.relabel_nodes(graph, name_mapping),
            node_values=self._node_values.merge(new_values),
        )

    def reduce(
        self,
        key: None | Hashable = None,
        *,
        index: None | Hashable = None,
        axis: None | int = None,
        name: None | Hashable = None,
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
            Integer axis index to reduce over. Only one of index and axis can be given.
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
        indices = _node_indices(key)
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
            raise ValueError(f"Node '{name}' already exists in the graph.")

        graph = self.graph.copy()
        name = MappedNode(name=name, indices=new_index) if new_index else name
        graph.add_node(name, **attrs)
        graph.add_edge(key, name)

        return Graph(graph, node_values=self._node_values)

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

    def to_networkx(self, value_attr: str = 'value') -> nx.DiGraph:
        """
        Convert to a NetworkX graph, spelling out the internal array structures as
        explicit nodes.

        Parameters
        ----------
        value_attr:
            The name of the attribute on nodes that holds the array-like object.
        """
        graph = self.graph
        for index_name, index in reversed(self.indices.items()):
            # Find all nodes with this index
            nodes = [
                node
                for node in graph.nodes()
                if index_name
                in _node_indices(node.name if isinstance(node, NodeName) else node)
            ]
            # Make a copy for each index value
            graphs = [
                _rename_successors(
                    graph, successors=nodes, index=IndexValues.from_tuple(i)
                )
                for i in _yield_index([(index_name, index)])
            ]
            graph = nx.compose_all(graphs)
        # Replace all MappingNodes with their name
        new_names = {
            node: NodeName(node.name.name, node.index)
            for node in graph
            if isinstance(node, NodeName) and isinstance(node.name, MappedNode)
        }
        graph = nx.relabel_nodes(graph, new_names)

        # Get values using previously stored index values
        for node in graph.nodes:
            if (
                isinstance(node, NodeName)
                and (node_values := self._node_values.get(node.name)) is not None
            ):
                graph.nodes[node][value_attr] = node_values.sel(node.index.to_tuple())

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
        # Drop all node values that are not in the branch
        mapped = {a.name for a in ancestors if isinstance(a, MappedNode)}
        keep_values = [key for key in self._node_values.keys() if key in mapped]
        return Graph(
            self.graph.subgraph(ancestors),
            node_values=self._node_values.get_columns(keep_values),
        )

    def __setitem__(self, branch: Hashable | slice, other: Graph) -> None:
        """
        Set a new branch in place of the given branch.

        The new branch must have a unique sink node. The branch at `branch` is replaced
        with the new branch. The indices and node values are updated accordingly. The
        edges to successors of the old branch are connected to the sink of the new
        branch.
        """
        if isinstance(branch, slice):
            raise NotImplementedError('Setting slice not supported yet.')
        if not isinstance(other, Graph):
            raise TypeError(f'Expected {Graph}, got {type(other)}')
        new_branch = other.graph
        sink = _get_unique_sink(new_branch)
        # In the future, we could support this if BOTH sink and branch are MappedNodes
        # with identical indices.
        if isinstance(sink, MappedNode) or isinstance(branch, MappedNode):
            raise NotImplementedError('Mapped nodes not supported yet in __setitem__')
        new_branch = nx.relabel_nodes(new_branch, {sink: branch})
        if branch in self.graph:
            graph = _remove_ancestors(self.graph, branch)
            graph.nodes[branch].clear()
        else:
            graph = self.graph

        intersection_nodes = set(graph.nodes) & set(new_branch.nodes) - {branch}

        for node in intersection_nodes:
            if graph.pred[node] != new_branch.pred[node]:
                raise ValueError(
                    f"Node inputs differ for node '{node}':\n"
                    f"  {graph.pred[node]}\n"
                    f"  {new_branch.pred[node]}\n"
                )
            if graph.nodes[node] != new_branch.nodes[node]:
                raise ValueError(f"Node data differs for node '{node}'")

        graph = nx.compose(graph, new_branch)

        # Delay setting graph until we know no step fails
        self._node_values = self._node_values.merge(other._node_values)
        self.graph = graph
