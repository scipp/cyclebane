# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pandas as pd
import pytest
import xarray as xr

from cyclebane.node_values import NodeValues, ValueArray


class TestNodeValues:
    def test_merge_new_nodes(self):
        """Test merging new nodes that don't exist in the original NodeValues."""
        # Create initial NodeValues
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0),
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0),
        }
        node_values = NodeValues(initial_values)

        # Create new value arrays to merge
        new_values = {
            'c': ValueArray.from_array_like([7, 8, 9], axis_zero=1),
            'd': ValueArray.from_array_like([10, 11, 12], axis_zero=1),
        }

        # Merge and verify
        merged = node_values.merge(new_values)

        assert len(merged) == 4
        assert 'a' in merged
        assert 'b' in merged
        assert 'c' in merged
        assert 'd' in merged

        # Original nodes should be unchanged
        assert merged['a'] == initial_values['a']
        assert merged['b'] == initial_values['b']

        # New nodes should be added
        assert merged['c'] == new_values['c']
        assert merged['d'] == new_values['d']

    def test_merge_existing_node_raises_error(self):
        """Test that merging a node that already exists raises ValueError."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Try to merge a node that already exists
        new_values = {'a': ValueArray.from_array_like([7, 8, 9], axis_zero=1)}

        with pytest.raises(ValueError, match="Node 'a' has already been mapped"):
            node_values.merge(new_values)

    def test_merge_conflicting_index_names_raises_error(self):
        """Test that merging with conflicting index names raises ValueError."""
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)  # uses dim_0
        }
        node_values = NodeValues(initial_values)

        # Create new value with conflicting index name
        new_values = {
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0)  # also uses dim_0
        }

        with pytest.raises(ValueError, match="Conflicting new index names"):
            node_values.merge(new_values)

    def test_merge_empty_mapping(self):
        """Test merging an empty mapping returns original NodeValues."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        merged = node_values.merge({})

        assert len(merged) == 1
        assert merged['a'] == initial_values['a']

    def test_merge_same_values_not_added(self):
        """Test that identical values are not added when merging."""
        value_array = ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        initial_values = {'a': value_array}
        node_values = NodeValues(initial_values)

        # Try to merge the same value (should be filtered out)
        new_values = {
            'b': value_array  # Same object
        }

        merged = node_values.merge(new_values)

        # Should still only have the original node since the value is identical
        assert len(merged) == 1
        assert 'a' in merged
        assert 'b' not in merged

    def test_merge_with_pandas_series(self):
        """Test merging with pandas Series."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Create pandas Series with different index name
        series = pd.Series([4, 5, 6], name='custom_index')
        new_values = {'b': ValueArray.from_array_like(series, axis_zero=1)}

        merged = node_values.merge(new_values)

        assert len(merged) == 2
        assert 'a' in merged
        assert 'b' in merged

    def test_merge_with_xarray_dataarray(self):
        """Test merging with xarray DataArray."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Create xarray DataArray with different dimension name
        da = xr.DataArray([4, 5, 6], dims=['x'])
        new_values = {'b': ValueArray.from_array_like(da, axis_zero=1)}

        merged = node_values.merge(new_values)

        assert len(merged) == 2
        assert 'a' in merged
        assert 'b' in merged

    def test_merge_preserves_original_indices(self):
        """Test that merging preserves original indices."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)
        original_indices = node_values.indices

        new_values = {'b': ValueArray.from_array_like([4, 5, 6], axis_zero=1)}

        merged = node_values.merge(new_values)

        # Original indices should still be present
        for name, index in original_indices.items():
            assert name in merged.indices
            assert list(merged.indices[name]) == list(index)
