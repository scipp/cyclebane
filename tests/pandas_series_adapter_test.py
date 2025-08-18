# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pandas as pd
import pytest

from cyclebane.value_array_adapters import PandasSeriesAdapter


@pytest.fixture
def series() -> pd.Series:
    df = pd.DataFrame(
        {
            'material': ['A', 'A', 'B', 'A', 'C', 'C'],
            'sample': ['a', 'b', 'c', 'd', 'e', 'f'],
        }
    ).set_index('sample')
    return df['material']


class TestPandasSeriesAdapter:
    def test_adapter_from_grouping_row(self, series):
        adapter = PandasSeriesAdapter(series)
        assert adapter.shape == (6,)
        assert adapter.index_names == ('sample',)
        assert adapter.sel((('sample', 'c'),)) == 'B'

    def test_group_returns_multi_index_like_series(self, series):
        base_adapter = PandasSeriesAdapter(series)
        adapter = base_adapter.group()
        assert adapter.index_names == ('material', 'sample')
        assert adapter.shape == (3, 6)
        indices = adapter.indices
        assert list(indices['material']) == ['A', 'B', 'C']
        assert list(indices['sample']) == ['a', 'b', 'c', 'd', 'e', 'f']
        assert adapter.sel((('sample', 'c'),)) == 'B'
        assert adapter.sel((('material', 'A'),)) == 'A'
