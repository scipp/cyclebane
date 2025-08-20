# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

IndexName = Hashable
IndexValue = Hashable

T = TypeVar('T', bound='ValueArray')


class ValueArray(ABC):
    """
    Abstract base class for a series of values with an index that can be sliced.

    Used by :py:class:`NodeValues` to store the values of a given node in a graph. The
    abstraction allows for the use of different data structures to store the values of
    nodes in a graph, such as pandas.DataFrame, xarray.DataArray, numpy.ndarray, or
    simple Python iterables.
    """

    _registry: ClassVar = []

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
    def try_from(obj: Any, *, axis_zero: int = 0) -> ValueArray | None: ...

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._equal(other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    @abstractmethod
    def _equal(self: T, other: T) -> bool: ...

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

    def group(self, index_name: Hashable) -> ValueArray:
        """
        Group the values by their indices.

        This method is expected to return a new ValueArray that groups the values by
        their indices, allowing for operations like aggregation or summarization.
        """
        raise NotImplementedError(
            'ValueArray.group() is only implemented for Pandas series.'
        )

    def get_grouping(self) -> Grouping | None:
        """
        If the instance holds grouping information, return it.

        Meant to be overridden by subclasses that support grouping.
        """
        return None


@dataclass
class Grouping:
    indices: Iterable[Iterable[IndexValue]]
    index_name: IndexName
    group_index_name: IndexName
