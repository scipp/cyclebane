# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import Any, TypeVar

from . import value_array_adapters  # noqa: F401
from .value_array import ValueArray

IndexName = Hashable
IndexValue = Hashable

T = TypeVar('T', bound='ValueArray')


class NodeValues(Mapping[Hashable, ValueArray]):
    """
    A collection of pandas.DataFrame-like objects with distinct indices.

    This is used by :py:class:`Graph` to store the values of nodes in a graph.
    """

    def __init__(self, values: Mapping[Any, ValueArray]):
        # Start with empty and use merge to validate compatibility
        self._values = {}
        if values:
            merged = self.merge(values)
            self._values = merged._values

    def __len__(self) -> int:
        """Return the number of columns."""
        return len(self._values)

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over the column names."""
        return iter(self._values)

    def __getitem__(self, key: Hashable) -> ValueArray:
        """Return the column with the given name."""
        return self._values[key]

    def __setitem__(self, key: Hashable, value_array: ValueArray) -> None:
        """Add a single value array, checking for conflicts."""
        # Check if the value array is identical to existing one
        existing_value = self._values.get(key)
        if existing_value is not None:
            if existing_value == value_array:
                return  # No change needed
            else:
                raise ValueError(f"Node '{key}' has already been mapped")

        # Check for index conflicts
        existing_indices = self.indices
        new_indices = value_array.indices
        conflicting_names = set(new_indices.keys()) & set(existing_indices.keys())
        for name in conflicting_names:
            existing_index_values = list(existing_indices[name])
            new_index_values = list(new_indices[name])
            if existing_index_values != new_index_values:
                raise ValueError(
                    f'Conflicting index values for index name "{name}" of {key}: '
                    f'existing {existing_index_values} vs new {new_index_values}'
                )

        # Add the new value array
        self._values[key] = value_array

    @staticmethod
    def from_mapping(
        values: Mapping[Hashable, Sequence[Any]], axis_zero: int
    ) -> NodeValues:
        """Construct from a mapping of node names to value sequences."""
        # graph.map(param_table)
        #   {sample: [s1,s2,s3], material: [Si,Ge,Si], param: [p1,p2,p3]}
        # graph.groupby(material)
        # -> merge two indices into multi-index
        #   {material: {Si: {sample:[s1,s3]}, Ge: {sample:[s2]}}
        #   {material: {Si: {sample:[s1,s3], param:[p1,p3]}, Ge: {sample:[s2], param:[p2]}}
        value_arrays = {
            key: ValueArray.from_array_like(value, axis_zero=axis_zero)
            for key, value in values.items()
        }
        shapes = {array.shape for array in value_arrays.values()}
        if len(shapes) > 1:
            raise ValueError(
                'All value sequences in a map operation must have the same shape. '
                'Use multiple map operations if necessary.'
            )
        return NodeValues(value_arrays)

    def merge(self, value_arrays: Mapping[Any, ValueArray]) -> NodeValues:
        # Always create a copy
        result = NodeValues(dict(self._values))
        for key, value_array in value_arrays.items():
            result[key] = value_array
        return result

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
