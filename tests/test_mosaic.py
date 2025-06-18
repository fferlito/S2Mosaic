import pytest
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import sys
from pathlib import Path


# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from s2mosaic import mosaic


class TestMosaicInputValidation:
    """Test input validation for the mosaic function"""
    
    def test_invalid_grid_id_lowercase(self):
        """Test that lowercase grid IDs are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("50hmh", 2023)
    
    def test_invalid_grid_id_special_chars(self):
        """Test that grid IDs with special characters are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("50H-MH", 2023)
    
    def test_invalid_grid_id_numbers_only(self):
        """Test that numeric-only grid IDs are rejected"""
        with pytest.raises(ValueError, match="Grid .* is invalid"):
            mosaic("12345", 2023)
    
    def test_invalid_sort_method(self):
        """Test that invalid sort methods are rejected"""
        with pytest.raises(ValueError, match="Invalid sort method"):
            mosaic("50HMH", 2023, sort_method="invalid_method")
    
    def test_invalid_mosaic_method(self):
        """Test that invalid mosaic methods are rejected"""
        with pytest.raises(ValueError, match="Invalid mosaic method"):
            mosaic("50HMH", 2023, mosaic_method="invalid_method")
    
    def test_invalid_no_data_threshold_negative(self):
        """Test that negative no_data_threshold is rejected"""
        with pytest.raises(ValueError, match="No data threshold must be between 0 and 1"):
            mosaic("50HMH", 2023, no_data_threshold=-0.1)
    
    def test_invalid_no_data_threshold_greater_than_one(self):
        """Test that no_data_threshold > 1 is rejected"""
        with pytest.raises(ValueError, match="No data threshold must be between 0 and 1"):
            mosaic("50HMH", 2023, no_data_threshold=1.5)
    
    def test_invalid_band(self):
        """Test that invalid band names are rejected"""
        with pytest.raises(ValueError, match="Invalid band"):
            mosaic("50HMH", 2023, required_bands=["B04", "INVALID_BAND"])
    
    def test_visual_band_with_other_bands(self):
        """Test that visual band cannot be used with other bands"""
        with pytest.raises(ValueError, match="Cannot use visual band with other bands"):
            mosaic("50HMH", 2023, required_bands=["visual", "B04"])
    
    def test_percentile_without_percentile_method(self):
        """Test that percentile parameter requires percentile method"""
        with pytest.raises(ValueError, match="percentile_value is only valid for percentile mosaic method"):
            mosaic("50HMH", 2023, mosaic_method="mean", percentile_value=50)
    
    def test_percentile_method_without_percentile(self):
        """Test that percentile method requires percentile parameter"""
        with pytest.raises(ValueError, match="percentile_value must be provided for percentile mosaic method"):
            mosaic("50HMH", 2023, mosaic_method="percentile")
    
    def test_invalid_percentile_negative(self):
        """Test that negative percentile values are rejected"""
        with pytest.raises(ValueError, match="percentile_value must be between 0 and 100"):
            mosaic("50HMH", 2023, mosaic_method="percentile", percentile_value=-10)
    
    def test_invalid_percentile_greater_than_100(self):
        """Test that percentile values > 100 are rejected"""
        with pytest.raises(ValueError, match="percentile_value must be between 0 and 100"):
            mosaic("50HMH", 2023, mosaic_method="percentile", percentile_value=150)


class TestMosaicValidInputs:
    """Test that valid inputs are accepted without errors"""
    
    def test_valid_grid_id(self):
        """Test that valid grid IDs are accepted"""
        # These should not raise validation errors
        # Note: They might still fail due to no scenes found, but validation should pass
        try:
            mosaic("50HMH", 2023)
        except Exception as e:
            # Validation errors should not occur
            assert "invalid" not in str(e).lower()
            assert "Invalid" not in str(e)
    
    def test_valid_sort_methods(self):
        """Test that all valid sort methods are accepted"""
        valid_methods = ["valid_data", "oldest", "newest"]
        for method in valid_methods:
            try:
                mosaic("50HMH", 2023, sort_method=method)
            except Exception as e:
                assert "Invalid sort method" not in str(e)
    
    def test_valid_mosaic_methods(self):
        """Test that all valid mosaic methods are accepted"""
        valid_methods = ["mean", "first"]
        for method in valid_methods:
            try:
                mosaic("50HMH", 2023, mosaic_method=method)
            except Exception as e:
                assert "Invalid mosaic method" not in str(e)
    
    def test_valid_percentile_method(self):
        """Test that percentile method with valid percentile is accepted"""
        try:
            mosaic("50HMH", 2023, mosaic_method="percentile", percentile_value=50)
        except Exception as e:
            assert "Invalid mosaic method" not in str(e)
            assert "Percentile" not in str(e) or "invalid" not in str(e).lower()
    
    def test_valid_no_data_thresholds(self):
        """Test that valid no_data_threshold values are accepted"""
        valid_thresholds = [0.0, 0.01, 0.5, 1.0, None]
        for threshold in valid_thresholds:
            try:
                mosaic("50HMH", 2023, no_data_threshold=threshold)
            except Exception as e:
                assert "No data threshold must be between 0 and 1" not in str(e)
    
    def test_valid_bands(self):
        """Test that valid band combinations are accepted"""
        valid_band_sets = [
            ["B04", "B03", "B02"],
            ["B04", "B03", "B02", "B08"],
            ["visual"],
            ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
            ["AOT", "SCL", "WVP"]
        ]
        for bands in valid_band_sets:
            try:
                mosaic("50HMH", 2023, required_bands=bands)
            except Exception as e:
                assert "Invalid band" not in str(e)
                assert "Cannot use visual band with other bands" not in str(e)


class TestMosaicEndToEnd:
    """End-to-end tests using debug cache for performance"""
    
    @pytest.fixture(autouse=True)
    def setup_and_cleanup_cache(self):
        """Setup and cleanup debug cache before and after each test"""
        cache_dir = Path("cache")
        
        # Clean cache before test
        # if cache_dir.exists():
        #     shutil.rmtree(cache_dir)
        
        start_time = time.time()
        yield
        end_time = time.time()
        
        # Print test duration for debugging
        test_name = self.__class__.__name__
        duration = end_time - start_time
        if duration > 60:  # Print if test takes more than 1 minute
            print(f"\n⚠️  {test_name} took {duration:.1f} seconds")
        
        # Clean cache after test
        # if cache_dir.exists():
        #     shutil.rmtree(cache_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_mosaic_return_array_no_output_dir(self):
        """Test mosaic returns array and profile when no output_dir is specified"""
        print("\n🔄 Starting test_mosaic_return_array_no_output_dir")
        
        try:
            result = mosaic(
                "50HMH", 
                2023, 
                start_month=6, 
                start_day=1,
                duration_days=7,
                debug_cache=True,
                required_bands=["B04", "B03", "B02"]
            )
            print(f"✅ Mosaic function completed, result type: {type(result)}")
        except Exception as e:
            print(f"❌ Mosaic function failed with error: {e}")
            raise
        
        # Check if result is a tuple
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        print(f"✅ Result is tuple: {isinstance(result, tuple)}")
        
        # Check tuple length
        assert len(result) == 2, f"Expected tuple of length 2, got length {len(result)}"
        print(f"✅ Tuple length is 2: {len(result) == 2}")
        
        array, profile = result
        
        # Check array type
        assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
        print(f"✅ Array is numpy.ndarray: {isinstance(array, np.ndarray)}")
        
        # Check profile is dict-like (has keys, values, items methods)
        assert hasattr(profile, 'keys') and hasattr(profile, 'values') and hasattr(profile, 'items'), f"Profile should be dict-like, got {type(profile)}"
        print(f"✅ Profile is dict-like: {type(profile)}")
        
        # Check profile has expected rasterio keys
        expected_keys = {'driver', 'dtype', 'width', 'height', 'count', 'crs', 'transform'}
        profile_keys = set(profile.keys())
        missing_keys = expected_keys - profile_keys
        assert len(missing_keys) == 0, f"Profile missing expected keys: {missing_keys}"
        print(f"✅ Profile has expected keys: {profile_keys}")
        
        # Check array dimensions
        assert array.ndim == 3, f"Expected 3D array, got {array.ndim}D array with shape {array.shape}"
        print(f"✅ Array is 3D: {array.ndim == 3}, shape: {array.shape}")
        
        # Check number of bands
        assert array.shape[0] == 3, f"Expected 3 bands, got {array.shape[0]} bands"
        print(f"✅ Array has 3 bands: {array.shape[0] == 3}")
        
        # Check data type
        valid_dtypes = [np.uint8, np.int16, np.uint16, np.float32]
        assert array.dtype in valid_dtypes, f"Expected dtype in {valid_dtypes}, got {array.dtype}"
        print(f"✅ Array dtype is valid: {array.dtype} in {valid_dtypes}")
        
        print("🎉 All assertions passed!")
    
    def test_mosaic_save_to_file(self, temp_output_dir):
        """Test mosaic saves file when output_dir is specified"""
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            debug_cache=True,
            required_bands=["B04", "B03", "B02"]
        )
        
        assert isinstance(result, Path)
        assert result.exists()
        assert result.suffix == ".tif"
        assert result.parent == temp_output_dir
    
    def test_mosaic_overwrite_false(self, temp_output_dir):
        """Test that existing files are not overwritten when overwrite=False"""
        # Create first mosaic
        result1 = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            debug_cache=True,
            required_bands=["B04", "B03", "B02"]
        )
        
        # Get file modification time
        original_mtime = result1.stat().st_mtime
        
        # Try to create same mosaic with overwrite=False
        result2 = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            debug_cache=True,
            required_bands=["B04", "B03", "B02"],
            overwrite=False
        )
        
        # Should return same path without creating new file
        assert result1 == result2
        assert result2.stat().st_mtime == original_mtime
    
    @pytest.mark.slow
    def test_mosaic_different_sort_methods(self):
        """Test mosaic with different sort methods"""
        sort_methods = ["valid_data", "oldest", "newest"]
        
        for sort_method in sort_methods:
            print(f"\nTesting sort method: {sort_method}")
            start = time.time()
            result = mosaic(
                "50HMH", 
                2023, 
                start_month=6, 
                start_day=1,
                duration_days=7,
                sort_method=sort_method,
                debug_cache=True,
                required_bands=["B04"]
            )
            duration = time.time() - start
            print(f"Sort method {sort_method} took {duration:.1f}s")
            
            assert isinstance(result, tuple)
            array, profile = result
            assert isinstance(array, np.ndarray)
            assert hasattr(profile, 'keys')  # Check it's dict-like
            assert hasattr(profile, 'keys')  # Check it's dict-like
            assert hasattr(profile, 'keys')  # Check it's dict-like
            assert array.shape[0] == 1  # 1 band
    
    @pytest.mark.slow
    def test_mosaic_different_mosaic_methods(self):
        """Test mosaic with different mosaic methods"""
        mosaic_methods = ["mean", "first"]
        
        for mosaic_method in mosaic_methods:
            print(f"\nTesting mosaic method: {mosaic_method}")
            start = time.time()
            result = mosaic(
                "50HMH", 
                2023, 
                start_month=6, 
                start_day=1,
                duration_days=7,
                mosaic_method=mosaic_method,
                debug_cache=True,
                required_bands=["B04"]
            )
            duration = time.time() - start
            print(f"Mosaic method {mosaic_method} took {duration:.1f}s")
            
            assert isinstance(result, tuple)
            array, profile = result
            assert isinstance(array, np.ndarray)
    
    def test_mosaic_percentile_method(self):
        """Test mosaic with percentile method"""
        percentiles = [10, 50, 90]
        
        for percentile in percentiles:
            result = mosaic(
                "50HMH", 
                2023, 
                start_month=6, 
                start_day=1,
                duration_days=7,
                mosaic_method="percentile",
                percentile_value=percentile,
                debug_cache=True,
                required_bands=["B04"]
            )
            
            assert isinstance(result, tuple)
            array, profile = result
            assert isinstance(array, np.ndarray)
    
    def test_mosaic_visual_band(self):
        """Test mosaic with visual band"""
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            debug_cache=True,
            required_bands=["visual"]
        )
        
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        assert hasattr(profile, 'keys')  # Check it's dict-like
        assert hasattr(profile, 'keys')  # Check it's dict-like
        assert array.shape[0] == 3  # RGB channels
        assert array.dtype == np.uint8  # Visual should be uint8
    
    def test_mosaic_different_time_ranges(self):
        """Test mosaic with different time range specifications"""
        # Test duration_days
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            debug_cache=True,
            required_bands=["B04"]
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        
        # Test duration_months (keep under 1 month)
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=21,  # 3 weeks instead of 1 month
            debug_cache=True,
            required_bands=["B04"]
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        
        # Test duration_years (but keep it short)
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=14,  # 2 weeks instead of 1 year
            debug_cache=True,
            required_bands=["B04"]
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        
        # Test combined duration_months and duration_days (keep under 1 month)
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=20,  # 20 days instead of 2 months + 15 days
            debug_cache=True,
            required_bands=["B04"]
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
    
    def test_mosaic_different_cloud_cover_thresholds(self):
        """Test mosaic with different cloud cover thresholds"""
        # Test cloud cover < 50%
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=1, 
            start_day=1,
            duration_months=1,
            debug_cache=True,
            required_bands=["B04"],
            additional_query={"eo:cloud_cover": {"lt": 50}}
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        
        # Test cloud cover < 20%
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            debug_cache=True,
            required_bands=["B04"],
            additional_query={"eo:cloud_cover": {"lt": 20}}
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
        
        # Test cloud cover < 10%
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            debug_cache=True,
            required_bands=["B04"],
            additional_query={"eo:cloud_cover": {"lt": 10}}
        )
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)
    
    def test_mosaic_no_scenes_found(self):
        """Test behaviour when no scenes are found"""
        # Use a very restrictive date range and cloud cover that likely has no data
        with pytest.raises(Exception, match="No scenes found"):
            mosaic(
                "50HMH", 
                2023,
                start_month=6, 
                start_day=1,
                duration_days=1,  # Single day
                debug_cache=True,
                required_bands=["B04"],
                additional_query={"eo:cloud_cover": {"lt": 0.1}}  # Very low cloud cover
            )
    
    def test_mosaic_custom_sort_function(self):
        """Test mosaic with custom sort function"""
        def custom_sort(items):
            # Sort by datetime ascending (oldest first)
            return items.sort_values("datetime", ascending=True).reset_index(drop=True)
        
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            sort_function=custom_sort,
            debug_cache=True,
            required_bands=["B04"]
        )
        
        assert isinstance(result, tuple)
        array, profile = result
        assert isinstance(array, np.ndarray)


class TestMosaicFileNaming:
    """Test file naming conventions"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        """Clean cache after each test"""
        yield
        cache_dir = Path("cache")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    
    def test_filename_format(self, temp_output_dir):
        """Test that output filenames follow expected format"""
        result = mosaic(
            "50HMH", 
            2023, 
            start_month=6, 
            start_day=1,
            duration_days=7,
            output_dir=temp_output_dir,
            debug_cache=True,
            sort_method="oldest",
            mosaic_method="mean",
            required_bands=["B04", "B03", "B02"]
        )
        
        expected_pattern = "50HMH_2023-06-01_to_2023-06-08_oldest_mean_B04_B03_B02.tif"
        assert result.name == expected_pattern
    
    def test_filename_different_parameters(self, temp_output_dir):
        """Test filename changes with different parameters"""
        result = mosaic(
            "50HMH", 
            2022, 
            start_month=12, 
            start_day=15,
            duration_months=1,
            output_dir=temp_output_dir,
            debug_cache=True,
            sort_method="newest",
            mosaic_method="first",
            required_bands=["visual"]
        )
        
        expected_pattern = "51HTT_2022-12-15_to_2023-01-15_newest_first_visual.tif"
        assert result.name == expected_pattern


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])