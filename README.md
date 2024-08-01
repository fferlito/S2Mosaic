**# S2Mosaic 🛰️🌍**

S2Mosaic is a Python package for creating cloud-free mosaics from Sentinel-2 satellite imagery. It allows users to generate composite images for specified grid areas and time ranges, with various options for scene selection and mosaic creation.

**## Features 🌟**

- Create Sentinel-2 mosaics for specific grid areas and time ranges
- Flexible scene selection methods: by valid data percentage, oldest, or newest scenes
- Multiple mosaic creation methods: mean or first valid pixel
- Support for different spectral bands, including visual (RGB) composites
- State-of-the-art cloud masking using the OmniCloudMask library
- Export mosaics as GeoTIFF files or return as NumPy arrays

**## Note 📝**

We use OmniCloudMask for state-of-the-art cloud and cloud shadow masking. OCM will run significantly faster if an available NVIDIA GPU is present.

**## Installation 🛠️**

You can install S2Mosaic using pip:
```
pip install s2mosaic
```

**## Usage Example 1 🚀**

Here's a basic example of how to use S2Mosaic:

```python
from s2mosaic import mosaic
from pathlib import Path

# Create a mosaic for a specific grid area and time range
result = mosaic(
    grid_id="50HMH", # Sentinel-2 scene grid ID
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2, # Duration to collect data from
    output_dir=Path("output"), # Output directory for mosaic TIFF files
    sort_method="valid_data", # Method to sort potential scenes before download
    mosaic_method="mean", # Approach used to combine scenes
    required_bands=['visual'], # Required Sentinel-2 bands
    no_data_threshold=0.001 # Threshold for early stopping
)

print(f"Mosaic saved to: {result}")
```

This example creates a mosaic for the grid area "50HMH" for the first two months of 2022, using the visual (TCI) product. The scenes are sorted by valid data percentage, and the mosaic is created using the mean of valid pixels. The process stops iterating through scenes once the no_data_threshold is reached.

**## Usage Example 2 🔬**

Here's another example of how to use S2Mosaic:

```python
from s2mosaic import mosaic

# Create a mosaic for a specific grid area and time range
array, rio_profile = mosaic(
    grid_id="50HMH",
    start_year=2022,
    start_month=1,
    start_day=1,
    duration_months=2,
    sort_method="valid_data",
    mosaic_method="mean",
    required_bands=["B04", "B03", "B02", "B08"],
    no_data_threshold=0.001
)

print(f"Mosaic array shape: {array.shape}")
```

Similar to the example above but with 16-bit red, green, blue, and NIR bands returned as a NumPy array and rasterio profile. 📊

**## Advanced Usage 🧠**

S2Mosaic provides several options for customizing the mosaic creation process:

- `sort_method`: Choose between "valid_data", "oldest", or "newest" to determine scene selection priority.
- `mosaic_method`: Use "mean" for an average of valid pixels or "first" to use the first valid pixel.
- `required_bands`: Specify which spectral bands to include in the mosaic. Use ["visual"] for an RGB composite.
- `no_data_threshold`: Set the threshold for considering a pixel as no-data. Set to None to process all scenes.
- `ocm_batch_size`: Set the batch size for OmniCloudMask inference (default: 6).
- `ocm_inference_dtype`: Set the data type for OmniCloudMask inference (default: "bf16").

For more detailed information on these options and additional functionality, please refer to the function docstring in the source code.

**## Contributing 🤝**

Contributions to S2Mosaic are welcome! Please feel free to submit pull requests, create issues, or suggest improvements. 🙌

**## License 📄**

This project is licensed under the MIT License. ⚖️

**## Acknowledgments 🙏**

This package uses the Planetary Computer STAC API and the OmniCloudMask library for cloud masking.