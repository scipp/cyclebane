# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import abc
from typing import Any, Hashable, Iterable, Mapping, Sequence

IndexName = Hashable
IndexValue = Hashable


class ValueArray(ABC):
    """Abstract base class for a series of values with an index that can be sliced."""

    @staticmethod
    def from_array_like(values: Any, *, axis_zero: int = 0) -> ValueArray:
        if hasattr(values, 'dims'):
            return DataArrayAdapter(values)
        if values.__class__.__name__ == 'ndarray':
            return NumpyArrayAdapter(values, axis_zero=axis_zero)
        return IterableAdapter(values, index=range(len(values)), axis_zero=axis_zero)

    @abstractmethod
    def sel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        """Return data by selecting from index with given name and index value."""

    @abstractmethod
    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> ValueArray:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def index_names(self) -> tuple[IndexName, ...]:
        pass

    @property
    @abstractmethod
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        pass


class PandasSeriesAdapter(ValueArray):
    def __init__(self, series, *, axis_zero: int = 0):
        self._series = series
        self._axis_zero = axis_zero

    def sel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        if len(key) != 1:
            raise ValueError('PandasSeriesAdapter only supports single index')
        _, i = key[0]
        return self._series.loc[i]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> PandasSeriesAdapter:
        return PandasSeriesAdapter(self._series[key], axis_zero=self._axis_zero)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._series),)

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self.indices)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        name = (
            self._series.index.name
            if self._series.index.name is not None
            else f'dim_{self._axis_zero}'
        )
        return {name: self._series.index}


class DataArrayAdapter(ValueArray):
    def __init__(
        self,
        data_array,
    ):
        self._data_array = data_array

    def sel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        # Note: Eventually we will want to distinguish between dims without coords,
        # where we will use a range index, and dims with coords, where we will use the
        # coord as an index. For now everything is a range index.
        # We use isel because of sel, since we default to range indices for now.
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
        # TODO Cannot be range after slicing! This is currently inconsistent with how
        # NumPy and iterable adapters work.
        return {name: range(size) for name, size in zip(self.index_names, self.shape)}


class NumpyArrayAdapter(ValueArray):
    def __init__(
        self,
        array,
        *,
        indices: dict[IndexName, Iterable[IndexValue]] | None = None,
        axis_zero: int = 0,
    ):
        import numpy as np

        self._array = np.asarray(array)
        if indices is None:
            indices = {
                f'dim_{i+axis_zero}': range(size)
                for i, size in enumerate(self._array.shape)
            }
        self._indices = indices
        self._axis_zero = axis_zero

    def sel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        index_tuple = tuple(self._indices[k].index(i) for k, i in key)
        return self._array[index_tuple]

    def __getitem__(self, key: int | slice) -> NumpyArrayAdapter:
        if isinstance(key, tuple):
            raise NotImplementedError('Cannot select from multi-dim value array')
        return NumpyArrayAdapter(
            self._array[key],
            indices={
                index_name: index_values[key]
                for index_name, index_values in self._indices.items()
            },
            axis_zero=self._axis_zero,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(f'dim_{i+self._axis_zero}' for i in range(self._array.ndim))

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return self._indices


class IterableAdapter(ValueArray):
    def __init__(
        self, values: Sequence[Any], *, index: Iterable[IndexValue], axis_zero: int = 0
    ):
        self._values = values
        self._index = index
        self._axis_zero = axis_zero

    def sel(self, key: list[tuple[IndexName, IndexValue]]) -> Any:
        if len(key) != 1:
            raise ValueError('IterableAdapter only supports single index')
        _, i = key[0]
        return self._values[self._index.index(i)]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> IterableAdapter:
        return IterableAdapter(
            self._values[key], index=self._index[key], axis_zero=self._axis_zero
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._values),)

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return (f'dim_{self._axis_zero}',)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return {f'dim_{self._axis_zero}': self._index}


class NodeValues(abc.Mapping[Hashable, ValueArray]):
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
                    f'{tuple(self.indices)}'
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
