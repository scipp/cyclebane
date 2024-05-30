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

    _registry = []

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        ValueArray._registry.append(cls)

    @staticmethod
    def from_array_like(values: Any, *, axis_zero: int = 0) -> ValueArray:
        # Reversed to ensure SequenceAdapter is tried last, as it is the most general
        # SequenceAdapter is defined right after this class so it is registered first
        for subclass in reversed(ValueArray._registry):
            if (a := subclass.try_from(values, axis_zero=axis_zero)) is not None:
                return a
        raise ValueError(f'Cannot create ValueArray from {values}')

    @staticmethod
    @abstractmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> ValueArray | None:
        ...

    @abstractmethod
    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        """Return data by selecting from index with given name and index value."""

    def loc(self, key: dict[IndexName, slice]) -> ValueArray:
        if not all(isinstance(i, slice) for i in key.values()):
            raise ValueError('ValueArray.loc only accepts slices, not integers')
        if not set(key).issubset(set(self.index_names)):
            raise ValueError(
                f'ValueArray.loc got {key.keys()}, not a subset of {self.index_names}'
            )
        return self[key]

    @abstractmethod
    def __getitem__(self, key: dict[IndexName, slice]) -> ValueArray:
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

    @staticmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> SequenceAdapter | None:
        return SequenceAdapter(obj, axis_zero=axis_zero)

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        if len(key) != 1:
            raise ValueError('SequenceAdapter only supports single index')
        _, i = key[0]
        return self._values[self._index.index(i)]

    def __getitem__(self, key: dict[IndexName, slice]) -> SequenceAdapter:
        _, i = next(iter(key.items()))
        return SequenceAdapter(
            self._values[i], index=self._index[i], axis_zero=self._axis_zero
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


class PandasSeriesAdapter(ValueArray):
    def __init__(self, series: 'pandas.Series', *, axis_zero: int = 0):
        self._series = series
        self._axis_zero = axis_zero

    @staticmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> PandasSeriesAdapter | None:
        try:
            import pandas
        except ModuleNotFoundError:
            return None
        if isinstance(obj, pandas.Series):
            return PandasSeriesAdapter(obj, axis_zero=axis_zero)

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

    def __getitem__(self, key: dict[IndexName, slice]) -> PandasSeriesAdapter:
        _, i = next(iter(key.items()))
        return PandasSeriesAdapter(self._series[i], axis_zero=self._axis_zero)

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


class XarrayDataArrayAdapter(ValueArray):
    def __init__(
        self,
        data_array: 'xarray.DataArray',
    ):
        default_indices = {
            dim: range(size)
            for dim, size in data_array.sizes.items()
            if dim not in data_array.coords
        }
        self._data_array = data_array.assign_coords(default_indices)

    @staticmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> XarrayDataArrayAdapter | None:
        try:
            import xarray

            if isinstance(obj, xarray.DataArray):
                return XarrayDataArrayAdapter(obj)
        except ModuleNotFoundError:
            pass

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        return self._data_array.sel(dict(key))

    def __getitem__(self, key: dict[IndexName, slice]) -> XarrayDataArrayAdapter:
        return XarrayDataArrayAdapter(self._data_array.isel(key))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self._data_array.dims)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return {
            dim: self._data_array.coords[dim].values for dim in self._data_array.dims
        }


class ScippDataArrayAdapter(ValueArray):
    def __init__(self, data_array: 'scipp.DataArray'):
        import scipp

        default_indices = {
            dim: scipp.arange(dim, size, unit=None)
            for dim, size in data_array.sizes.items()
            if dim not in data_array.coords
        }
        self._data_array = data_array.assign_coords(default_indices)

    @staticmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> ScippDataArrayAdapter | None:
        try:
            import scipp

            if isinstance(obj, scipp.Variable):
                return ScippDataArrayAdapter(scipp.DataArray(obj))
            if isinstance(obj, scipp.DataArray):
                return ScippDataArrayAdapter(obj)
        except ModuleNotFoundError:
            pass

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        import scipp

        values = self._data_array
        for dim, value in key:
            # Reconstruct label, to use label-based indexing instead of positional
            if isinstance(value, tuple):
                value, unit = value
            else:
                unit = None
            label = scipp.scalar(value, unit=unit)
            # Scipp indexing uses a comma to separate dimension label from the index,
            # unlike Numpy and other libraries where it separates the indices for
            # different axes.
            values = values[dim, label]
        return values

    def __getitem__(self, key: dict[IndexName, slice]) -> ScippDataArrayAdapter:
        values = self._data_array
        for dim, i in key:
            values = values[dim, i]
        return ScippDataArrayAdapter(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data_array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self._data_array.dims)

    def _index_for_dim(self, dim: str) -> list[tuple[Any, 'scipp.Unit']]:
        # Work around some NetworkX errors. Probably scipp.Variable lacks functionality.
        # For now we return a list of tuples, where the first element is the value and
        # the second is the unit.
        coord = self._data_array.coords[dim]
        unit = coord.unit
        if unit is None:
            return coord.values
        unit = str(unit)
        return [(value, unit) for value in coord.values]

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return {dim: self._index_for_dim(dim) for dim in self._data_array.dims}


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

    @staticmethod
    def try_from(obj: Any, *, axis_zero: int = 0) -> NumpyArrayAdapter | None:
        try:
            import numpy
        except ModuleNotFoundError:
            return None
        if isinstance(obj, numpy.ndarray):
            return NumpyArrayAdapter(obj, axis_zero=axis_zero)

    def sel(self, key: tuple[tuple[IndexName, IndexValue], ...]) -> Any:
        index_tuple = tuple(self._indices[k].index(i) for k, i in key)
        return self._array[index_tuple]

    def __getitem__(self, key: dict[IndexName, slice]) -> NumpyArrayAdapter:
        return NumpyArrayAdapter(
            self._array[tuple(key.get(k, slice(None)) for k in self._indices)],
            indices={
                index_name: (index_values[key.get(index_name, slice(None))])
                for index_name, index_values in self._indices.items()
            },
            axis_zero=self._axis_zero,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def index_names(self) -> tuple[IndexName, ...]:
        return tuple(self._indices)

    @property
    def indices(self) -> dict[IndexName, Iterable[IndexValue]]:
        return self._indices


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
