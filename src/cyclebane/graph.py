# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import networkx as nx

from .node_values import IndexName, IndexValue, NodeValues
from .value_array import Grouping


def _get_unique_sink(graph: nx.DiGraph) -> Hashable:
    sink_nodes = [node for node in graph.nodes if graph.out_degree(node) == 0]
    if len(sink_nodes) != 1:
        raise ValueError(f'Graph must have exactly one sink node, got {sink_nodes}')
    return sink_nodes[0]


def _labeled_key(graph: nx.DiGraph, key: Hashable, match_index: Hashable) -> Hashable:
    """Find the node for ``key`` in a labeled graph, by original name if mapped."""
    if key in graph:
        return key
    matches = [
        node
        for node in graph.nodes
        if isinstance(node, MappedNode)
        and node.name == key
        and match_index in node.indices
    ]
    if len(matches) == 0:
        raise KeyError(f"Node '{key}' does not exist in the graph.")
    if len(matches) > 1:
        raise KeyError(f"Node '{key}' is ambiguous. Found {matches}.")
    return matches[0]


def _get_new_node_name(graph: nx.DiGraph) -> str:
    while True:
        name = str(uuid4())
        if name not in graph:
            return name


def _remove_ancestors(graph: nx.DiGraph, node: Hashable) -> nx.DiGraph:
    """
    Returns a copy of the graph without the ancestors of the given node.

    Warning: Returns the original graph if the node has no data and ancestors.
    """
    ancestors = nx.ancestors(graph, node)
    if not ancestors and not graph.nodes[node]:
        return graph
    graph_without_node = graph.copy()
    graph_without_node.remove_node(node)
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
    graph.nodes[node].clear()
    return graph


@dataclass(frozen=True, slots=True)
class IndexValues:
    """
    Index values used as part of :py:class:`NodeName`.

    Conceptually, this is a mapping from index names to index values.
    """

    axes: tuple[IndexName, ...]
    values: tuple[IndexValue, ...]

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


@dataclass(frozen=True, slots=True)
class _ReduceSpec:
    """Records what a reduce node consumes; applied when deriving node indices.

    The index names to drop are resolved when ``reduce`` is called, so that
    indices added by later ``map`` calls flow through the reduce node.
    """

    drop: frozenset[IndexName]
    extra_index_name: None | IndexName


def _node_name(node: Hashable) -> Hashable:
    if isinstance(node, MappedNode):
        return node.name
    return node


def _node_indices(node: Hashable) -> tuple[IndexName, ...]:
    if isinstance(node, MappedNode):
        return node.indices
    return ()


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
        return Graph(
            self.graph.graph,
            node_values=node_values,
            reductions=self.graph._reductions,
        )


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

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        node_values: NodeValues | None = None,
        reductions: dict[Hashable, _ReduceSpec] | None = None,
    ):
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
        reductions:
            A mapping from reduce-node names to reduce specs. Internal, do not use.
        """
        self.graph = graph
        self._node_values = node_values or NodeValues({})
        self._reductions = dict(reductions or {})

    def copy(self) -> Graph:
        return Graph(
            self.graph.copy(),
            node_values=self._node_values,
            reductions=self._reductions,
        )

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

        The mapped source nodes and their successors gain an index (dimension).
        This only records the values and indices; nodes are spelled out into one
        copy per index value in :py:meth:`to_networkx`, with values set as an
        attribute on the source-node copies.

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

        for root in new_values:
            if graph.in_degree(root) > 0:
                raise ValueError(f"Mapped node '{root}' is not a source node")

        # Note that the graph is not relabeled: which nodes carry which indices is
        # derived from the mapped roots and reachability in _derive_indices, at the
        # time it is needed. This keeps node names stable and makes `map` commute
        # with adding branches to the graph.
        return Graph(
            graph,
            node_values=self._node_values.merge(new_values),
            reductions=self._reductions,
        )

    def _derive_indices(self) -> dict[Hashable, tuple[IndexName, ...]]:
        """Derive the indices carried by each node from mapped roots and reduces.

        A node carries the indices of the mapped roots that reach it, minus what
        reduce nodes on the way consume. The per-node index order matches what
        incremental relabeling used to produce: indices of later ``map`` calls
        first, order within one call preserved; groupby's extra index appended
        last. Nodes carrying no indices are absent from the result.
        """
        if not self._node_values and not self._reductions:
            # Fast path; also keeps graphs with cycles (which some users allow
            # until compute time) working as long as nothing is mapped.
            return {}
        graph = self.graph
        root_blocks = {
            root: arr.index_names
            for root, arr in self._node_values.items()
            if arr.get_grouping() is None and root in graph
        }
        priority: dict[IndexName, tuple[int, int]] = {}
        for i, block in enumerate(root_blocks.values()):
            for pos, name in enumerate(block):
                priority.setdefault(name, (-i, pos))

        names: dict[Hashable, set[IndexName]] = {}
        extras: dict[Hashable, tuple[IndexName, ...]] = {}
        result: dict[Hashable, tuple[IndexName, ...]] = {}
        for node in nx.topological_sort(graph):
            current: set[IndexName] = set()
            extra: tuple[IndexName, ...] = ()
            for pred in graph.predecessors(node):
                current |= names[pred]
                for name in extras[pred]:
                    if name not in extra:
                        extra = (*extra, name)
            if (root_block := root_blocks.get(node)) is not None:
                current |= set(root_block)
            if (spec := self._reductions.get(node)) is not None:
                current -= spec.drop
                extra = tuple(name for name in extra if name not in spec.drop)
                if spec.extra_index_name is not None:
                    extra = (*extra, spec.extra_index_name)
            names[node] = current
            extras[node] = extra
            if full := tuple(sorted(current, key=priority.__getitem__)) + extra:
                result[node] = full
        return result

    def _labeled_graph(self) -> nx.DiGraph:
        """Return a copy of the graph with index-carrying nodes relabeled as
        :py:class:`MappedNode`, the representation :py:meth:`to_networkx` works on."""
        mapping = {
            node: MappedNode(name=node, indices=indices)
            for node, indices in self._derive_indices().items()
        }
        return nx.relabel_nodes(self.graph, mapping, copy=True)

    def node_indices(self, key: Hashable) -> tuple[IndexName, ...]:
        """Return the index names carried by the given node, () if unmapped."""
        return self._derive_indices().get(key, ())

    @property
    def value_keys(self) -> tuple[Hashable, ...]:
        """Names of the nodes that have associated values.

        Contains the mapped source nodes, and the reduce nodes of groupby
        operations (which store the grouping).
        """
        return tuple(self._node_values)

    def groupby(self, node: Hashable) -> GroupbyGraph:
        return GroupbyGraph(
            self.graph,
            node_values=self._node_values,
            node=node,
            reductions=self._reductions,
        )

    def reduce(
        self,
        key: None | Hashable = None,
        *,
        index: None | Hashable = None,
        axis: None | int = None,
        name: None | Hashable = None,
        attrs: None | dict[str, Any] = None,
        _extra_index_name: None | IndexName = None,
    ) -> Graph:
        """
        Reduce over the given index or axis previously created with :py:meth:`map`.

        If neither index nor axis is given, all axes are reduced.

        Parameters
        ----------
        key:
            The name of the node to reduce. If not given, tries to find a unique
            sink node.
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
        if key not in self.graph:
            raise KeyError(f"Node '{key}' does not exist in the graph.")
        # Resolve what is reduced into concrete index names now; indices added
        # by later `map` calls are unaffected and flow through the reduce node.
        indices = self.node_indices(key)
        if index is not None:
            if index not in indices:
                raise ValueError(f"Node '{key}' does not have index '{index}'.")
            drop = frozenset({index})
        elif axis is not None:
            # TODO We can support indexing from the back in the future.
            if axis < 0 or axis >= len(indices):
                raise ValueError(f"Node '{key}' does not have axis '{axis}'.")
            drop = frozenset({indices[axis]})
        else:
            drop = frozenset(indices)

        if name in self.graph:
            raise ValueError(f"Node '{name}' already exists in the graph.")

        graph = self.graph.copy()
        graph.add_node(name, **attrs)
        graph.add_edge(key, name)

        reductions = dict(self._reductions)
        reductions[name] = _ReduceSpec(drop=drop, extra_index_name=_extra_index_name)
        return Graph(graph, node_values=self._node_values, reductions=reductions)

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
        # Nodes carrying indices are stored under their plain names; relabel them
        # as MappedNode based on the derived indices before spelling out.
        graph = self._labeled_graph()

        # Maintain a list of actual node values, without groupings, since we only want
        # to set the former (user-provided) on (input) nodes.
        node_values = self._node_values.copy()
        groupby_graphs = []
        # Handle groupby/reduce operations. The regular iterative node duplication does
        # not work in this case. We have to handle the graph edges that correspond to
        # a particular groupby/reduce operation in isolation, or else we get broken
        # result in the presence of multiple (chained or not) groupby operations. The
        # resulting graphs that correspond to the grouping are later composed with the
        # rest of the graph.
        for key, values in self._node_values.items():
            if (grouping := values.get_grouping()) is not None:
                del node_values[key]
                key = _labeled_key(graph, key, match_index=grouping.group_index_name)
                # Note there should be only a single predecessor for the grouping node.
                groupby_graph = graph.subgraph([*graph.predecessors(key), key]).copy()
                # Remove edges, or the loop for the regular map/reduce will add
                # all-to-all edges between these nodes
                graph.remove_edges_from(groupby_graph.edges)
                groupby_graphs.append(self._make_groupby_graph(grouping, groupby_graph))

        # Handle regular map/reduce operations
        for index_name, index in reversed(self.indices.items()):
            graphs = _clone_graph(graph, index_name, index)
            graph = nx.compose_all(graphs)

        if groupby_graphs:
            graph = nx.compose_all([*groupby_graphs, graph])

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
                and (value_array := node_values.get(node.name)) is not None
            ):
                graph.nodes[node][value_attr] = value_array.sel(node.index.to_tuple())

        return graph

    def _make_groupby_graph(
        self, grouping: Grouping, groupby_graph: nx.DiGraph
    ) -> nx.DiGraph:
        for index_name, index in reversed(self.indices.items()):
            if index_name == grouping.index_name:
                continue
            graphs = _clone_graph(groupby_graph, index_name, index)
            if index_name == grouping.group_index_name:
                subgraphs = [
                    _clone_graph(group_graph, grouping.index_name, idx)
                    for idx, group_graph in zip(grouping.indices, graphs, strict=True)
                ]
                # Flatten nested list of graphs
                graphs = [g for sublist in subgraphs for g in sublist]
            groupby_graph = nx.compose_all(graphs)
        return groupby_graph

    def __getitem__(self, key: Hashable | slice) -> Graph:
        """
        Get the branch of the graph rooted at the given node.

        The branch is a subgraph containing the given node and all its ancestors.
        Think of this like a Git branch, where the given node is the head of the branch.
        """
        if isinstance(key, slice):
            raise NotImplementedError('Only single nodes are supported ')
        if key not in self.graph:
            raise KeyError(f"Node '{key}' does not exist in the graph.")
        ancestors = nx.ancestors(self.graph, key)
        ancestors.add(key)
        # Drop all node values that are not in the branch
        keep_values = [key for key in self._node_values.keys() if key in ancestors]
        return Graph(
            self.graph.subgraph(ancestors),
            node_values=self._node_values.get_columns(keep_values),
            reductions={k: v for k, v in self._reductions.items() if k in ancestors},
        )

    def __delitem__(self, key: Hashable | slice) -> None:
        """
        Delete the branch of the graph rooted at the given node.
        """
        if isinstance(key, slice):
            raise NotImplementedError('Only single nodes are supported ')
        if key not in self.graph:
            raise KeyError(f"Node '{key}' does not exist in the graph.")
        if self.node_indices(key):
            # Not clear what to do in this case, as it would leave a lot of mapped
            # nodes without a source that could provide data.
            raise ValueError('Cannot delete mapped node.')
        graph = _remove_ancestors(self.graph, key)
        keep_values = [key for key in self._node_values.keys() if key in graph]
        self._node_values = self._node_values.get_columns(keep_values)
        self._reductions = {k: v for k, v in self._reductions.items() if k in graph}
        self.graph = graph

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
        # Replacing an existing branch must not change whether it is mapped, as
        # this would silently change the indices of its dependents. Setting a
        # new branch is fine either way; its indices follow from derivation.
        if branch in self.graph and bool(other.node_indices(sink)) != bool(
            self.node_indices(branch)
        ):
            raise NotImplementedError(
                'Trying to set mapped node on non-mapped node (or vice versa) is not '
                'possible in __setitem__'
            )
        if branch in new_branch and branch != sink:
            # Renaming the sink to the branch name would silently merge it with
            # the like-named node inside the new branch. This typically means
            # the new branch computes the branch node from itself (e.g. a
            # reduction of a mapped branch assigned back to the same name);
            # rename the node inside the new branch to make this well-defined.
            raise ValueError(
                f"Cannot set branch '{branch}': the new branch already contains "
                f"a node of that name. Use a distinct name for the node inside "
                "the new branch."
            )
        new_branch = nx.relabel_nodes(new_branch, {sink: branch})
        if branch in self.graph:
            graph = _remove_ancestors(self.graph, branch)
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
        reductions = dict(self._reductions)
        # A reduce spec of a replaced branch node must not survive replacement.
        reductions.pop(branch, None)
        reductions.update(other._reductions)
        if sink in reductions and sink != branch:
            # The sink was renamed to the branch name above.
            reductions[branch] = reductions.pop(sink)
        self._reductions = reductions

        # Ensure we preserve the node values of the branch, if it exists. This step is
        # necessary since __setitem__ effectively renames the sink node of the input
        # graph to the branch name.
        if sink in self._node_values:
            node_values = self._node_values[sink]
            del self._node_values[sink]
            self._node_values[branch] = node_values

        self.graph = graph


class GroupbyGraph:
    """
    A graph that has been grouped by a specific index.

    This is a specialized graph that is used to represent the result of a groupby
    operation on a Cyclebane graph. It allows for operations on the grouped data,
    such as aggregation or summarization.
    """

    # TODO Should we support a custom new dim name here, instead of using `node`?
    def __init__(
        self,
        graph: nx.DiGraph,
        node_values: NodeValues,
        node: Hashable,
        reductions: dict[Hashable, _ReduceSpec] | None = None,
    ):
        self._graph = graph
        self._node_values = node_values
        self._reductions = dict(reductions or {})
        values_to_group_by = node_values[node]
        self._group_index_name = node
        self._index_name = values_to_group_by.index_names[0]
        self._groups = values_to_group_by.group(index_name=node)

    # TODO Require specifying index!?
    def reduce(
        self,
        key: None | Hashable = None,
        *,
        name: None | Hashable = None,
        attrs: None | dict[str, Any] = None,
    ) -> Graph:
        """
        Reduce the grouped graph over the given index or axis.

        Parameters
        ----------
        key:
            The name of the node to reduce. If not given, tries to find a unique
            sink node.
        name:
            The name of the new node. If not given, a unique name is generated.
        attrs:
            Attributes to set on the new node(s).
        """
        # Generate name here since we want to store grouping on the new "reduce" node.
        name = name or _get_new_node_name(self._graph)
        # Why do we store the grouping here? This works well with existing mechanisms,
        # e.g., __getitem__, which needs to decided what subset of node values to keep
        # when returning a subgraph.
        node_values = self._node_values.merge({name: self._groups})
        graph = Graph(self._graph, node_values=node_values, reductions=self._reductions)
        return graph.reduce(
            key=key,
            index=self._index_name,
            name=name,
            attrs=attrs,
            _extra_index_name=self._group_index_name,
        )


def _clone_graph(
    graph: nx.DiGraph, index_name: IndexName, index: Iterable[IndexValue]
) -> list[nx.DiGraph]:
    # Find all nodes with this index
    nodes = [
        node
        for node in graph.nodes()
        if index_name
        in _node_indices(node.name if isinstance(node, NodeName) else node)
    ]
    # Make a copy for each index value
    return [
        _rename_successors(
            graph, successors=nodes, index=IndexValues((index_name,), (i,))
        )
        for i in index
    ]
