# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import abc
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    import numpy
    import pandas
    import scipp
    import xarray

IndexName = Hashable
IndexValue = Hashable


class ValueArray(ABC):
    """
    Abstract base class for a series of values with an index that can be sliced.

    Used by :py:class:`NodeValues` to store the values of a given node in a graph. The
    abstraction allows for the use of different data structures to store the values of
    nodes in a graph, such as pandas.DataFrame, xarray.DataArray, numpy.ndarray, or
    simple Python iterables.
    """

    @staticmethod
    def from_array_like(values: Any, *, axis_zero: int = 0) -> ValueArray:
        if hasattr(values, 'dims'):
            return DataArrayAdapter(values)
        if values.__class__.__name__ == 'ndarray':
            return NumpyArrayAdapter(values, axis_zero=axis_zero)
        return SequenceAdapter(values, axis_zero=axis_zero)

    @abstractmethod
    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
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
    def __init__(self, series: 'pandas.Series', *, axis_zero: int = 0):
        self._series = series
        self._axis_zero = axis_zero

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        if len(key) != 1:
            raise ValueError('PandasSeriesAdapter only supports single index')
        index_name, i = key[0]
        if index_name != self.index_names[0]:
            raise ValueError(
                f'Unexpected index name {index_name} for PandasSeriesAdapter with '
                f'index names {self.index_names}'
            )
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
        index_name = (
            self._series.index.name
            if self._series.index.name is not None
            else f'dim_{self._axis_zero}'
        )
        return (index_name,)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return {self.index_names[0]: self._series.index}


class DataArrayAdapter(ValueArray):
    def __init__(
        self,
        data_array: 'xarray.DataArray | scipp.Variable | scipp.DataArray',
    ):
        self._data_array = data_array

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        # Note: Eventually we will want to distinguish between dims without coords,
        # where we will use a range index, and dims with coords, where we will use the
        # coord as an index. For now everything is a range index.
        # We use isel instead of sel, since we default to range indices for now.
        if hasattr(self._data_array, 'isel'):
            return self._data_array.isel(dict(key))
        values = self._data_array
        for label, i in key:
            # This is Scipp notation, Xarray uses the 'isel' method. Scipp indexing
            # uses a comma to separate dimension label from the index, unlike Numpy
            # and other libraries where it separates the indices for different axes.
            values = values[label, i]
        return values

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> DataArrayAdapter:
        # We have not implemented slicing that correctly handles the range-index setup
        # in, e.g., self.indices. This is a placeholder implementation.
        raise NotImplementedError('DataArrayAdapter does not support slicing')
        # If we insert range indices as coords, this implementation will work.
        return DataArrayAdapter(self._data_array[key])

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self._data_array.dims)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return {name: range(size) for name, size in zip(self.index_names, self.shape)}


class NumpyArrayAdapter(ValueArray):
    def __init__(
        self,
        array: 'numpy.ndarray',
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

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        index_tuple = tuple(self._indices[k].index(i) for k, i in key)
        return self._array[index_tuple]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> NumpyArrayAdapter:
        if isinstance(key, tuple):
            raise NotImplementedError('Cannot select from multi-dim value array')
        if isinstance(key, int):
            # This would break current handling of axis naming.
            raise NotImplementedError('Cannot select single value from value array')
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


class SequenceAdapter(ValueArray):
    def __init__(
        self,
        values: Sequence[Any],
        *,
        index: Iterable[IndexValue] | None = None,
        axis_zero: int = 0,
    ):
        self._values = values
        self._index = index or range(len(values))
        self._axis_zero = axis_zero

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        if len(key) != 1:
            raise ValueError('SequenceAdapter only supports single index')
        _, i = key[0]
        return self._values[self._index.index(i)]

    def __getitem__(
        self, key: int | slice | tuple[int | slice, ...]
    ) -> SequenceAdapter:
        if isinstance(key, tuple) and len(key) > 1:
            raise ValueError('SequenceAdapter is always 1-D')
        return SequenceAdapter(
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
    """
    A collection of pandas.DataFrame-like objects with distinct indices.

    This is used by :py:class:`Graph` to store the values of nodes in a graph.
    """

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

    @staticmethod
    def from_mapping(
        values: Mapping[Hashable, Sequence[Any]], axis_zero: int
    ) -> NodeValues:
        """Construct from a mapping of node names to value sequences."""
        keys = tuple(values)
        if (columns := getattr(values, 'columns', None)) is not None:
            value_arrays = {
                key: PandasSeriesAdapter(values.iloc[:, i], axis_zero=axis_zero)
                for key, i in zip(keys, range(len(columns)))
            }
        else:
            value_arrays = {
                key: ValueArray.from_array_like(values[key], axis_zero=axis_zero)
                for key in keys
            }
            shapes = {array.shape for array in value_arrays.values()}
            if len(shapes) > 1:
                raise ValueError(
                    'All value sequences in a map operation must have the same shape. '
                    'Use multiple map operations if necessary.'
                )
        return NodeValues(value_arrays)

    def merge(self, value_arrays: Mapping[Hashable, ValueArray]) -> NodeValues:
        if value_arrays:
            named = next(iter(value_arrays.values())).index_names
            if any([name in self.indices for name in named]):
                raise ValueError(
                    f'Conflicting new index names {named} with existing '
                    f'{tuple(self.indices)}'
                )
        for node in value_arrays:
            if node in self:
                raise ValueError(f"Node '{node}' has already been mapped")
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
