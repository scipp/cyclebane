# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections import abc
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


@dataclass(frozen=True)
class IndexValues:
    """
    Index values used as part of :py:class:`NodeName`.

    Conceptually, this is a mapping from index names to index values.
    """

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
    """Node name with indices used for mapped nodes when converting to NetworkX."""

    name: Hashable
    index: IndexValues

    def merge_index(self, other: IndexValues) -> NodeName:
        return NodeName(name=self.name, index=self.index.merge_index(other))

    def __str__(self):
        return f'{self.name}({self.index})'


@dataclass(frozen=True)
class MappedNode:
    """
    Key for a node in :py:class:`Graph` representing a collection of "mapped" nodes.
    """

    name: Hashable
    indices: tuple[IndexName, ...]


def node_with_indices(node: Hashable, indices: tuple[IndexName, ...]) -> MappedNode:
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
        if graph.in_degree(root) > 0:
            raise ValueError(f"Mapped node '{root}' is not a source node")
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

        # TODO all broken here?
        # TODO having to slice indices and values independently shows design problem.
        # Why can't we just keep, e.g., the DataFrames and op on those?
        # Can Graph.indices be computed dynamically?
        out.indices = {
            name: slice_index(name, index) for name, index in self.graph.indices.items()
        }
        out._node_values = {
            index_names: slice_values(index_names, values)
            for index_names, values in self.graph._node_values.items()
        }
        return out


MappingToArrayLike = Any  # dict[str, Numpy|DataArray], DataFrame, etc.


class ValueArray:
    """A series of values with an index that can be sliced."""

    @staticmethod
    def from_array_like(values: Any, *, axis_zero: int = 0) -> ValueArray:
        if hasattr(values, 'dims'):
            return DataArrayAdapter(values)
        return NumpyArrayAdapter(values, axis_zero=axis_zero)

    def isel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        pass

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> ValueArray:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        pass

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        pass


class PandasSeriesAdapter(ValueArray):
    def __init__(self, series, *, axis_zero: int = 0):
        self._series = series
        self._axis_zero = axis_zero

    def isel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        if len(key) != 1:
            raise ValueError('PandasSeriesAdapter only supports single index')
        _, i = key[0]
        return self._series.loc[i]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> PandasSeriesAdapter:
        return PandasSeriesAdapter(self._series[key])

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._series),)

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return (self._series.index.name,)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        name = (
            self._series.index.name
            if self._series.index.name is not None
            else f'dim_{self._axis_zero}'
        )
        return {name: self._series.index}


class DataArrayAdapter(ValueArray):
    def __init__(self, data_array):
        self._data_array = data_array

    def isel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        if hasattr(self._data_array, 'isel'):
            return self._data_array.isel(dict(key))
        values = self._data_array
        for label, i in key:
            # This is Scipp notation, Xarray uses the 'isel' method.
            values = values[(label, i)]
        return values

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> DataArrayAdapter:
        return DataArrayAdapter(self._data_array[key])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self._data_array.dims)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        # TODO Cannot be range after slicing!
        return {name: range(size) for name, size in zip(self.index_names, self.shape)}


class NumpyArrayAdapter(ValueArray):
    def __init__(self, array, *, axis_zero: int = 0):
        import numpy as np

        self._array = np.asarray(array)
        self._axis_zero = axis_zero

    def isel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        return self._array[tuple(i for _, i in key)]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> NumpyArrayAdapter:
        return NumpyArrayAdapter(self._array[key], axis_zero=self._axis_zero)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(f'dim_{i+self._axis_zero}' for i in range(self._array.ndim))

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        # TODO Cannot be range after slicing!
        return {name: range(size) for name, size in zip(self.index_names, self.shape)}


# TODO add adapter class performing the logic of _get_indices, abstracting differences
# between DataFrame, DataArray, ndarray, ..., such that NodeValues can operate on the
# adapter class on a common interface
class NodeValues(abc.Mapping):
    """A collection of pandas.DataFrame-like objects with distinct indices."""

    def __init__(self, values: Mapping[Hashable, ValueArray]):
        self._values = values

    def __len__(self) -> int:
        """Return the number of columns."""
        return len(self._values)

    def __iter__(self) -> Iterable[Hashable]:
        """Iterate over the column names."""
        return iter(self._values)

    def __getitem__(self, key: Hashable) -> ValueArray:
        """Return the column with the given name."""
        return self._values[key]

    def _to_value_arrays(
        self, values: Mapping[Hashable, Sequence[Any]]
    ) -> Mapping[Hashable, ValueArray]:
        keys = tuple(values)
        ndim = len(self.indices)
        if (columns := getattr(values, 'columns', None)) is not None:
            return {
                key: PandasSeriesAdapter(values.iloc[:, i], axis_zero=ndim)
                for key, i in zip(keys, range(len(columns)))
            }
        return {
            key: ValueArray.from_array_like(values[key], axis_zero=ndim) for key in keys
        }

    def merge_from_mapping(
        self, node_values: Mapping[Hashable, Sequence[Any]]
    ) -> NodeValues:
        """Append from a mapping of node names to value sequences."""
        for node in node_values:
            if any(node in mapping for mapping in self._values.keys()):
                raise ValueError(f"Node '{node}' has already been mapped")
        value_arrays = self._to_value_arrays(node_values)
        shapes = [array.shape for array in value_arrays.values()]
        if len(set(shapes)) != 1:
            raise ValueError(
                'All value sequences in a map operation must have the same shape. '
                'Use multiple map operations if necessary.'
            )
        return self.merge(value_arrays)

    def merge(self, value_arrays: Mapping[Hashable, ValueArray]) -> NodeValues:
        if value_arrays:
            named = next(iter(value_arrays.values())).index_names
            if any([name in self.indices for name in named]):
                raise ValueError(
                    f'Conflicting new index names {named} with existing '
                    f'{list(self.indices)}'
                )
        return NodeValues({**self._values, **value_arrays})

    def get_columns(self, keys: list[Hashable]) -> NodeValues:
        """Select a subset of columns."""
        return NodeValues({key: self._values[key] for key in keys})

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        """Return the indices of the NodeValues object."""
        value_indices = [value.indices for value in self._values.values()]
        return {
            name: index for indices in value_indices for name, index in indices.items()
        }


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
    The current implementation is a proof of concept, there is a number of things to
    improve:
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

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        node_values: NodeValues | None = None,
        value_attr: str = 'value',
    ):
        self.graph = graph
        self._value_attr = value_attr
        self._next_node_values = node_values or NodeValues({})

    def copy(self) -> Graph:
        return Graph(
            self.graph.copy(),
            node_values=self._next_node_values,
            value_attr=self._value_attr,
        )

    @property
    def value_attr(self) -> str:
        return self._value_attr

    @property
    def index_names(self) -> tuple[IndexName]:
        return tuple(self.indices)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return self._next_node_values.indices

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
        _next_node_values = self._next_node_values.merge_from_mapping(node_values)
        root_nodes = tuple(node_values.keys())
        named = tuple(
            idx
            for idx in _next_node_values.indices
            if idx not in self._next_node_values.indices
        )

        # Make sure root nodes exist in graph, add them if not. This choice allows for
        # mapping, e.g., with multiple columns from a DataFrame, representing labels
        # used later for groupby operations.
        root_node_graph = nx.DiGraph()
        root_node_graph.add_nodes_from(root_nodes)
        graph = nx.compose(self.graph, root_node_graph)

        successors = _find_successors(graph, root_nodes=root_nodes)
        name_mapping: dict[Hashable, MappedNode] = {}
        for node in successors:
            name_mapping[node] = node_with_indices(node, named)

        # TODO When removing nodes, e.g., via __getitem__, we should remove also the
        # node values. We need to make a shallow copy though, or extract columns and
        # store them individually. As we basically want do to indexing ops on the node
        # values we should consider using a dedicated class NodeValues to encapsulate
        # this complexity.
        return Graph(
            nx.relabel_nodes(graph, name_mapping),
            node_values=_next_node_values,
            value_attr=self.value_attr,
        )

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

        return Graph(
            graph, node_values=self._next_node_values, value_attr=self.value_attr
        )

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
        for name, col in self._next_node_values.items():
            for node in graph.nodes:
                if isinstance(node, NodeName) and node.name == name:
                    graph.nodes[node][self.value_attr] = col.isel(node.index.to_tuple())

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
        mapped = set(a.name for a in ancestors if isinstance(a, MappedNode))
        keep_values = [key for key in self._next_node_values.keys() if key in mapped]
        return Graph(
            self.graph.subgraph(ancestors),
            node_values=self._next_node_values.get_columns(keep_values),
            value_attr=self.value_attr,
        )

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
        # TODO This is not working yet as it should
        self._next_node_values = self._next_node_values.merge(other._next_node_values)
        self.graph = graph
