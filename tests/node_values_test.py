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

    def test_merge_compatible_indices_same_values(self):
        """Test merging with compatible indices and same index values."""
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0),  # uses dim_0
        }
        node_values = NodeValues(initial_values)

        # Create new value with same index name and same index values
        new_values = {
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0)  # also uses dim_0
        }

        # This should work since the indices are compatible
        merged = node_values.merge(new_values)

        assert len(merged) == 2
        assert 'a' in merged
        assert 'b' in merged

    def test_merge_compatible_indices_different_values_raises_error(self):
        """Test that merging with same index name but different values raises error."""
        initial_values = {
            'a': ValueArray.from_array_like(
                [1, 2, 3], axis_zero=0
            ),  # uses dim_0 with range(3)
        }
        node_values = NodeValues(initial_values)

        # Create new value with same index name but different length
        new_values = {
            'b': ValueArray.from_array_like(
                [4, 5], axis_zero=0
            )  # uses dim_0 with range(2)
        }

        with pytest.raises(ValueError, match="Conflicting new index names"):
            node_values.merge(new_values)

    def test_merge_identical_node_and_value_not_added(self):
        """Test that identical nodes with identical values are not added again."""
        value_array = ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        initial_values = {'a': value_array}
        node_values = NodeValues(initial_values)

        # Try to merge the same node with the same value
        new_values = {'a': value_array}

        merged = node_values.merge(new_values)

        # Should still only have the original node
        assert len(merged) == 1
        assert 'a' in merged
        assert merged['a'] == value_array

    def test_merge_existing_node_different_value_raises_error(self):
        """Test that merging existing node with different value raises error."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Try to merge same node with different value
        new_values = {'a': ValueArray.from_array_like([4, 5, 6], axis_zero=0)}

        with pytest.raises(ValueError, match="Node 'a' has already been mapped"):
            node_values.merge(new_values)

    def test_merge_xarray_same_dim_different_coords_should_fail(self):
        """Test merging xarray DataArrays with same dimension names but different coordinate values."""
        # Create initial xarray DataArray with time coordinate
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        initial_values = {'sensor1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with same dimension name but different coordinate values
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [40, 50, 60]})
        new_values = {'sensor2': ValueArray.from_array_like(da2)}

        # This should fail because time coordinates are different
        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            node_values.merge(new_values)

    def test_merge_xarray_same_dim_same_coords_should_succeed(self):
        """Test merging xarray DataArrays with same dim names and same coord values."""
        # Create initial xarray DataArray with time coordinate
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        initial_values = {'sensor1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with same dimension name and same coordinate values
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [10, 20, 30]})
        new_values = {'sensor2': ValueArray.from_array_like(da2)}

        # This should succeed because time coordinates are identical
        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert 'sensor1' in merged
        assert 'sensor2' in merged

    def test_merge_2d_xarray_partial_overlap_should_fail(self):
        """Test merging 2D xarray DataArrays with conflicting coords."""
        # Create initial 2D xarray DataArray
        da1 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['time', 'space'],
            coords={'time': [10, 20], 'space': ['A', 'B']},
        )
        initial_values = {'data1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with overlapping 'time' dimension but different values
        da2 = xr.DataArray([5, 6], dims=['time'], coords={'time': [30, 40]})
        new_values = {'data2': ValueArray.from_array_like(da2)}

        # This should fail because 'time' coordinates conflict
        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            node_values.merge(new_values)
