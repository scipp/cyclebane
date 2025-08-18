# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pandas as pd
import pytest
import xarray as xr

from cyclebane.node_values import NodeValues, ValueArray


class TestNodeValues:
    def test_init_with_compatible_indices(self):
        """Test that initialization succeeds with compatible indices."""
        values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0),  # dim_0: [0,1,2]
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0),  # dim_0: [0,1,2]
        }
        node_values = NodeValues(values)
        assert len(node_values) == 2
        assert set(node_values.keys()) == {'a', 'b'}

    def test_init_with_conflicting_indices(self):
        """Test that initialization fails with conflicting indices."""
        values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0),  # dim_0: [0,1,2]
            'b': ValueArray.from_array_like([4, 5], axis_zero=0),  # dim_0: [0,1]
        }
        with pytest.raises(
            ValueError, match='Conflicting index values for index name "dim_0"'
        ):
            NodeValues(values)

    def test_init_with_different_index_names(self):
        """Test that initialization succeeds with different index names."""
        values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0),  # dim_0
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=1),  # dim_1
        }
        node_values = NodeValues(values)
        assert len(node_values) == 2
        assert set(node_values.keys()) == {'a', 'b'}
        assert list(node_values.indices) == ['dim_0', 'dim_1']

    def test_init_with_empty_values(self):
        """Test that initialization succeeds with empty values."""
        node_values = NodeValues({})
        assert len(node_values) == 0

    def test_init_with_xarray_conflicting_coords(self):
        """Test that init fails with xarray DataArrays and conflicting coordinates."""
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [40, 50, 60]})

        values = {
            'sensor1': ValueArray.from_array_like(da1),
            'sensor2': ValueArray.from_array_like(da2),
        }

        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            NodeValues(values)

    def test_init_with_xarray_compatible_coords(self):
        """Test that init succeeds with xarray DataArrays and compatible coords."""
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [10, 20, 30]})

        values = {
            'sensor1': ValueArray.from_array_like(da1),
            'sensor2': ValueArray.from_array_like(da2),
        }

        node_values = NodeValues(values)
        assert len(node_values) == 2
        assert set(node_values.keys()) == {'sensor1', 'sensor2'}


class TestNodeValuesMerge:
    """Systematic testing of all merge scenarios."""

    def test_merge_new_nodes_no_index_conflicts(self):
        """Adding new nodes with completely different index names."""
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        }  # dim_0
        node_values = NodeValues(initial_values)

        new_values = {'b': ValueArray.from_array_like([4, 5, 6], axis_zero=1)}  # dim_1

        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert set(merged.keys()) == {'a', 'b'}

    def test_merge_new_nodes_same_index_names_same_index_values(self):
        """Adding new nodes with same index names and same index values."""
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        }  # dim_0: [0,1,2]
        node_values = NodeValues(initial_values)

        new_values = {
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0)
        }  # dim_0: [0,1,2]

        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert set(merged.keys()) == {'a', 'b'}

    def test_merge_new_nodes_same_index_names_different_values_raises(self):
        """Adding new nodes with same index names but different index values."""
        initial_values = {
            'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        }  # dim_0: [0,1,2]
        node_values = NodeValues(initial_values)

        new_values = {
            'b': ValueArray.from_array_like([4, 5], axis_zero=0)
        }  # dim_0: [0,1]

        with pytest.raises(
            ValueError, match='Conflicting index values for index name "dim_0"'
        ):
            node_values.merge(new_values)

    def test_merge_existing_node_identical_value(self):
        """Re-adding existing node with identical value (should be no-op)."""
        value_array = ValueArray.from_array_like([1, 2, 3], axis_zero=0)
        initial_values = {'a': value_array}
        node_values = NodeValues(initial_values)

        new_values = {'a': value_array}  # Same object

        merged = node_values.merge(new_values)
        assert len(merged) == 1
        assert merged is not node_values  # Should return new object (copy)
        assert merged['a'] is value_array  # But should contain same value array

    def test_merge_existing_node_equal_but_different_object(self):
        """Re-adding existing node with equal but different value object."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Create equivalent but different object
        new_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}

        merged = node_values.merge(new_values)
        assert len(merged) == 1
        assert merged is not node_values  # Should return new object (copy)

    def test_merge_existing_node_different_value(self):
        """Re-adding existing node with different value."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        new_values = {'a': ValueArray.from_array_like([4, 5, 6], axis_zero=0)}

        with pytest.raises(ValueError, match="Node 'a' has already been mapped"):
            node_values.merge(new_values)

    def test_merge_empty_new_values(self):
        """Merging empty mapping."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        merged = node_values.merge({})
        assert merged is not node_values  # Should return new object (copy)
        assert len(merged) == 1
        assert 'a' in merged

    def test_merge_empty_initial_values(self):
        """Merging into empty NodeValues."""
        node_values = NodeValues({})

        new_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}

        merged = node_values.merge(new_values)
        assert len(merged) == 1
        assert 'a' in merged

    def test_merge_multiple_new_nodes_mixed_conflicts(self):
        """Multiple new nodes where some conflict and some don't."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        new_values = {
            'b': ValueArray.from_array_like([4, 5, 6], axis_zero=0),  # Compatible
            'c': ValueArray.from_array_like(
                [7, 8], axis_zero=0
            ),  # Conflicts with dim_0
        }

        with pytest.raises(
            ValueError, match='Conflicting index values for index name "dim_0"'
        ):
            node_values.merge(new_values)

    def test_merge_multiple_new_nodes_one_existing(self):
        """Multiple new nodes where one already exists."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        new_values = {
            'a': ValueArray.from_array_like(
                [4, 5, 6], axis_zero=0
            ),  # Exists but different
            'b': ValueArray.from_array_like([7, 8, 9], axis_zero=1),  # New
        }

        with pytest.raises(ValueError, match="Node 'a' has already been mapped"):
            node_values.merge(new_values)

    def test_merge_partial_index_overlap_compatible(self):
        """New nodes with partial index overlap that is compatible."""
        # Initial: has 'time' and 'space' indices
        da1 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['time', 'space'],
            coords={'time': [10, 20], 'space': ['A', 'B']},
        )
        initial_values = {'data1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # New: only has 'time' index but with same values
        da2 = xr.DataArray([5, 6], dims=['time'], coords={'time': [10, 20]})
        new_values = {'data2': ValueArray.from_array_like(da2)}

        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert set(merged.keys()) == {'data1', 'data2'}

    def test_merge_partial_index_overlap_incompatible(self):
        """New nodes with partial index overlap that is incompatible."""
        # Initial: has 'time' and 'space' indices
        da1 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['time', 'space'],
            coords={'time': [10, 20], 'space': ['A', 'B']},
        )
        initial_values = {'data1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # New: has 'time' index but with different values
        da2 = xr.DataArray([5, 6], dims=['time'], coords={'time': [30, 40]})
        new_values = {'data2': ValueArray.from_array_like(da2)}

        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            node_values.merge(new_values)

    def test_merge_different_adapter_types_compatible(self):
        """Merging different ValueArray adapter types with compatible indices."""
        # Initial: SequenceAdapter with dim_0
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # New: PandasSeriesAdapter with same index values but different representation
        series = pd.Series([4, 5, 6], index=[0, 1, 2], name='dim_0')
        new_values = {'b': ValueArray.from_array_like(series)}

        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert set(merged.keys()) == {'a', 'b'}

    def test_merge_different_adapter_types_incompatible(self):
        """Merging different ValueArray adapter types with incompatible indices."""
        # Initial: SequenceAdapter with dim_0: [0,1,2]
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # New: PandasSeriesAdapter with different index values
        series = pd.Series([4, 5, 6], index=['x', 'y', 'z'], name='dim_0')
        new_values = {'b': ValueArray.from_array_like(series)}

        with pytest.raises(
            ValueError, match='Conflicting index values for index name "dim_0"'
        ):
            node_values.merge(new_values)

    def test_merge_order_independence(self):
        """Verify that merge order doesn't matter for valid operations."""
        base_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(base_values)

        values1 = {'b': ValueArray.from_array_like([4, 5, 6], axis_zero=1)}
        values2 = {'c': ValueArray.from_array_like([7, 8, 9], axis_zero=2)}

        # Merge in one order
        result1 = node_values.merge(values1).merge(values2)

        # Merge in different order
        result2 = node_values.merge(values2).merge(values1)

        assert len(result1) == len(result2) == 3
        assert set(result1.keys()) == set(result2.keys()) == {'a', 'b', 'c'}

    def test_merge_pandas_series_custom_index_name(self):
        """Merging with pandas Series having custom index name."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Create pandas Series with different index name
        series = pd.Series([4, 5, 6], name='custom_index')
        new_values = {'b': ValueArray.from_array_like(series, axis_zero=1)}

        merged = node_values.merge(new_values)

        assert len(merged) == 2
        assert 'a' in merged
        assert 'b' in merged

    def test_merge_xarray_dataarray_custom_dimension_name(self):
        """Merging with xarray DataArray having custom dimension name."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)

        # Create xarray DataArray with different dimension name
        da = xr.DataArray([4, 5, 6], dims=['x'])
        new_values = {'b': ValueArray.from_array_like(da, axis_zero=1)}

        merged = node_values.merge(new_values)

        assert len(merged) == 2
        assert 'a' in merged
        assert 'b' in merged

    def test_merge_xarray_same_dim_different_coords_should_fail(self):
        """Merging xarray DataArrays with same dim names but different coord values."""
        # Create initial xarray DataArray with time coordinate
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        initial_values = {'sensor1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with same dim name but different coordinate values
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [40, 50, 60]})
        new_values = {'sensor2': ValueArray.from_array_like(da2)}

        # This should fail because time coordinates are different
        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            node_values.merge(new_values)

    def test_merge_xarray_same_dim_same_coords_should_succeed(self):
        """Merging xarray DataArrays with same dim names and same coord values."""
        # Create initial xarray DataArray with time coordinate
        da1 = xr.DataArray([1, 2, 3], dims=['time'], coords={'time': [10, 20, 30]})
        initial_values = {'sensor1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with same dim name and same coordinate values
        da2 = xr.DataArray([4, 5, 6], dims=['time'], coords={'time': [10, 20, 30]})
        new_values = {'sensor2': ValueArray.from_array_like(da2)}

        # This should succeed because time coordinates are identical
        merged = node_values.merge(new_values)
        assert len(merged) == 2
        assert 'sensor1' in merged
        assert 'sensor2' in merged

    def test_merge_2d_xarray_partial_overlap_should_fail(self):
        """Merging 2D xarray DataArrays with conflicting coords."""
        # Create initial 2D xarray DataArray
        da1 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=['time', 'space'],
            coords={'time': [10, 20], 'space': ['A', 'B']},
        )
        initial_values = {'data1': ValueArray.from_array_like(da1)}
        node_values = NodeValues(initial_values)

        # Create new xarray DataArray with overlapping 'time' dim but different values
        da2 = xr.DataArray([5, 6], dims=['time'], coords={'time': [30, 40]})
        new_values = {'data2': ValueArray.from_array_like(da2)}

        # This should fail because 'time' coordinates conflict
        with pytest.raises(
            ValueError, match='Conflicting index values for index name "time"'
        ):
            node_values.merge(new_values)

    def test_merge_preserves_original_indices(self):
        """Verify that merging preserves original indices."""
        initial_values = {'a': ValueArray.from_array_like([1, 2, 3], axis_zero=0)}
        node_values = NodeValues(initial_values)
        original_indices = node_values.indices

        new_values = {'b': ValueArray.from_array_like([4, 5, 6], axis_zero=1)}

        merged = node_values.merge(new_values)

        # Original indices should still be present
        for name, index in original_indices.items():
            assert name in merged.indices
            assert list(merged.indices[name]) == list(index)
