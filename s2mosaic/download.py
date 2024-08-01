from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, overload
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pkg_resources

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio as rio
import shapely
from pandas import DataFrame
from pystac.item_collection import ItemCollection
import pystac
from tqdm.auto import tqdm
from omnicloudmask import predict_from_array
import geopandas as gpd


def get_band(
    href: str, attempt: int = 0, res: int = 10
) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        singed_href = planetary_computer.sign(href)
        spatial_ratio = res / 10
        if "TCI_10m" in href:
            band_indexes = [1, 2, 3]
        else:
            band_indexes = [1]
        with rio.open(singed_href) as src:
            array = src.read(
                band_indexes,
                out_shape=(
                    len(band_indexes),
                    int(10980 / spatial_ratio),
                    int(10980 / spatial_ratio),
                ),
            ).astype(np.uint16)
            result = array, src.profile.copy()

            return result

    except Exception as e:
        print(e)
        print(f"Failed to open {href}")
        if attempt < 3:
            print(f"Trying again {attempt+1}")
            return get_band(href, attempt + 1)
        else:
            raise Exception(f"Failed to open {href}")


def ocm_cloud_mask(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
) -> np.ndarray:
    # download RG+NIR bands at 20m resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    get_band_20m = partial(get_band, res=20)

    hrefs = [item.assets[band].href for band in required_bands]

    with ThreadPoolExecutor(max_workers=len(required_bands)) as executor:
        bands_and_profiles = list(executor.map(get_band_20m, hrefs))

    # Separate bands and profiles
    bands, profiles = zip(*bands_and_profiles)
    mask = predict_from_array(
        input_array=np.vstack(bands),
        batch_size=batch_size,
        inference_dtype=inference_dtype,
    )[0]
    # interpolate mask back to 10m
    return mask.repeat(2, axis=0).repeat(2, axis=1) == 0


def format_progress(current, total, no_data_pct):
    return f"Scenes: {current}/{total} | No data: {no_data_pct:.2f}%"


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    s2_scene_size = 10980
    pixel_count = s2_scene_size * s2_scene_size
    if "visual" in required_bands:
        mosaic = np.zeros((3, s2_scene_size, s2_scene_size)).astype(np.float32)
    else:
        mosaic = np.zeros((len(required_bands), s2_scene_size, s2_scene_size)).astype(
            np.float32
        )

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size))

    pbar = tqdm(
        total=len(sorted_scenes),
        desc=format_progress(0, len(sorted_scenes), 100.0),
        leave=False,
        bar_format="{desc}",
    )
    get_bands_partial = partial(get_band, res=10)

    for index, item in enumerate(sorted_scenes["item"].tolist()):
        hrefs = [item.assets[band].href for band in required_bands]

        with ThreadPoolExecutor(max_workers=len(required_bands)) as executor:
            bands_and_profiles = list(executor.map(get_bands_partial, hrefs))

        bands, profiles = zip(*bands_and_profiles)

        if "visual" in required_bands:
            full_scene = np.array(bands[0])
        else:
            full_scene = np.vstack(bands)

        good_pixels = full_scene.sum(axis=0) > 0

        clear_pixels = ocm_cloud_mask(
            item=item,
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
        )
        combo_mask = clear_pixels * good_pixels
        good_pixel_tracker += combo_mask

        full_scene[:, ~combo_mask] = 0

        if mosaic_method == "mean":
            mosaic += full_scene
        elif mosaic_method == "first":
            new_valid_pixels = combo_mask & (mosaic == 0)
            mosaic[new_valid_pixels] = full_scene[new_valid_pixels]

        else:
            raise Exception("Invalid mosaic method, must be mean or first")

        no_data_sum = (good_pixel_tracker == 0).sum()
        no_data_pct = (no_data_sum / pixel_count) * 100
        pbar.set_description(
            format_progress(index + 1, len(sorted_scenes), no_data_pct)
        )
        # if using first or last method, stop if all pixels are filled
        if mosaic_method != "mean":
            if no_data_sum == 0:
                break
        # if no_data_threshold is set, stop if threshold is reached
        if no_data_threshold is not None:
            if no_data_sum < (pixel_count) * no_data_threshold:
                break
        pbar.update(1)

    remaining_scenes = pbar.total - pbar.n
    pbar.update(remaining_scenes)
    pbar.refresh()
    pbar.close()

    if mosaic_method == "mean":
        mosaic = mosaic / (good_pixel_tracker + 0.000001)
    if "visual" in required_bands:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.int16)

    return mosaic, profiles[-1]


def add_item_info(items: ItemCollection) -> DataFrame:
    """Split items by orbit and sort by no_data"""

    items_list = []
    for item in items:
        nodata = item.properties["s2:nodata_pixel_percentage"]
        data_pct = 100 - nodata

        cloud = item.properties["s2:high_proba_clouds_percentage"]
        shadow = item.properties["s2:cloud_shadow_percentage"]
        good_data_pct = data_pct * (1 - (cloud + shadow) / 100)
        capture_date = item.datetime

        items_list.append(
            {
                "item": item,
                "orbit": item.properties["sat:relative_orbit"],
                "good_data_pct": good_data_pct,
                "datetime": capture_date,
            }
        )

    items_df = pd.DataFrame(items_list)
    return items_df


def export_tif(
    array: np.ndarray,
    profile: Dict[str, Any],
    export_path: Path,
    required_bands: List[str],
) -> None:
    profile.update(count=array.shape[0], dtype=array.dtype, nodata=0, compress="lzw")
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = required_bands


def search_for_items(
    bounds, grid_id: str, start_date: date, end_date: date
) -> ItemCollection:
    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "query": {"s2:mgrs_tile": {"eq": grid_id}},
    }

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    return catalog.search(**query).item_collection()


def sort_items(items: DataFrame, sort_method: str) -> DataFrame:
    # Sort the dataframe by selected method then by orbit
    if sort_method == "valid_data":
        items_sorted = items.sort_values("good_data_pct", ascending=False)
        orbits = items_sorted["orbit"].unique()
        orbit_groups = {
            orbit: items_sorted[items_sorted["orbit"] == orbit] for orbit in orbits
        }

        result = []

        while any(len(group) > 0 for group in orbit_groups.values()):
            for orbit in orbits:
                if len(orbit_groups[orbit]) > 0:
                    result.append(orbit_groups[orbit].iloc[0])
                    orbit_groups[orbit] = orbit_groups[orbit].iloc[1:]

        items_sorted = pd.DataFrame(result).reset_index(drop=True)

    elif sort_method == "oldest":
        items_sorted = items.sort_values("datetime", ascending=True).reset_index(
            drop=True
        )
    elif sort_method == "newest":
        items_sorted = items.sort_values("datetime", ascending=False).reset_index(
            drop=True
        )
    else:
        raise Exception("Invalid sort method, must be valid_data, oldest or newest")

    return items_sorted


def get_extent_from_grid_id(grid_id: str) -> shapely.geometry.polygon.Polygon:
    S2_grid_file = Path(
        pkg_resources.resource_filename("s2mosaic", "S2_grid/sentinel_2_index.gpkg")
    )
    assert S2_grid_file.exists()
    S2_grid_gdf = gpd.read_file(S2_grid_file)
    try:
        S2_grid_gdf = S2_grid_gdf[S2_grid_gdf["Name"] == grid_id]
        if len(S2_grid_gdf) != 1:
            raise Exception(f"Grid {grid_id} not found")
        return S2_grid_gdf.iloc[0].geometry.buffer(-0.05)
    except Exception:
        raise Exception(f"Grid {grid_id} not found")


def define_dates(
    start_year: int,
    start_month: int,
    start_day: int,
    duration_years: int,
    duration_months: int,
    duration_days: int,
) -> Tuple[date, date]:
    start_date = datetime(start_year, start_month, start_day)
    end_date = start_date + relativedelta(
        years=duration_years, months=duration_months, days=duration_days
    )
    return start_date, end_date


SORT_VALID_DATA = "valid_data"
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"

VALID_SORT_METHODS = {SORT_VALID_DATA, SORT_OLDEST, SORT_NEWEST}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST}


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_threshold: Union[float, None],
    required_bands: List[str],
) -> None:
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Invalid sort method: {sort_method}. Must be one of {VALID_SORT_METHODS}"
        )
    if mosaic_method not in VALID_MOSAIC_METHODS:
        raise ValueError(
            f"Invalid mosaic method: {mosaic_method}. Must be one of {VALID_MOSAIC_METHODS}"
        )
    if no_data_threshold is not None:
        if not (0.0 <= no_data_threshold <= 1.0):
            raise ValueError(
                f"No data threshold must be between 0 and 1 or None, got {no_data_threshold}"
            )
    valid_bands = [
        "AOT",
        "SCL",
        "WVP",
        "visual",
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    for band in required_bands:
        if band not in valid_bands:
            raise ValueError(f"Invalid band: {band}, must be one of {valid_bands}")
    if "visual" in required_bands and len(required_bands) > 1:
        raise ValueError("Cannot use visual band with other bands, must be used alone")


def get_output_path(
    output_dir: Union[Path, str],
    grid_id: str,
    start_date: date,
    end_date: date,
    sort_method: str,
    mosaic_method: str,
    required_bands: List[str],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    bands_str = "_".join(required_bands)
    export_path = output_dir / (
        f"{grid_id}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_{sort_method}_{mosaic_method}_{bands_str}.tif"
    )
    return export_path


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: None = None,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
) -> Tuple[np.ndarray, Dict[str, Any]]: ...


@overload
def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Union[str, Path] = ...,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Optional[float] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
) -> Path: ...


def mosaic(
    grid_id: str,
    start_year: int,
    start_month: int = 1,
    start_day: int = 1,
    output_dir: Optional[Union[Path, str]] = None,
    sort_method: str = "valid_data",
    mosaic_method: str = "mean",
    duration_years: int = 0,
    duration_months: int = 0,
    duration_days: int = 0,
    required_bands: List[str] = ["B04", "B03", "B02", "B08"],
    no_data_threshold: Union[float, None] = 0.01,
    overwrite: bool = True,
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
) -> Union[Tuple[np.ndarray, Dict[str, Any]], Path]:
    """
    Create a Sentinel-2 mosaic for a specified grid and time range.

    This function generates a mosaic from Sentinel-2 satellite imagery based on the provided
    grid ID and time range. It can either return the mosaic data and metadata or save it as
    a GeoTIFF file.

    Args:
        grid_id (str): The ID of the grid area for which to create the mosaic (e.g., "50HMH").
        start_year (int): The start year of the time range.
        start_month (int, optional): The start month of the time range. Defaults to 1 (January).
        start_day (int, optional): The start day of the time range. Defaults to 1.
        output_dir (Optional[Union[Path, str]], optional): Directory to save the output GeoTIFF.
            If None, the mosaic is not saved to disk and is returned instead. Defaults to None.
        sort_method (str, optional): Method to sort scenes. Options are "valid_data", "oldest", or "newest". Defaults to "valid_data".
        mosaic_method (str, optional): Method to create the mosaic. Options are "mean" or "first". Defaults to "mean".
        duration_years (int, optional): Duration in years to add to the start date. Defaults to 0.
        duration_months (int, optional): Duration in months to add to the start date. Defaults to 0.
        duration_days (int, optional): Duration in days to add to the start date. Defaults to 0.
        required_bands (List[str], optional): List of required spectral bands.
            Defaults to ["B04", "B03", "B02", "B08"] (Red, Green, Blue, NIR).
        no_data_threshold (float, optional): Threshold for no data values. Defaults to 0.01.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        ocm_batch_size (int, optional): Batch size for OCM inference. Defaults to 6.
        ocm_inference_dtype (str, optional): Data type for OCM inference. Defaults to "bf16".

    Returns:
        Union[Tuple[np.ndarray, Dict[str, Any]], Path]: If output_dir is None, returns a tuple
        containing the mosaic array and metadata dictionary. If output_dir is provided,
        returns the path to the saved GeoTIFF file.

    Raises:
        Exception: If no scenes are found for the specified grid ID and time range.

    Note:
        - The function uses the STAC API to search for Sentinel-2 scenes.
        - If 'visual' is included in required_bands, it will be replaced with 'Red', 'Green', 'Blue' in the output.
        - The time range for scene selection is inclusive of the start date and exclusive of the end date.
    """
    bounds = get_extent_from_grid_id(grid_id)

    validate_inputs(sort_method, mosaic_method, no_data_threshold, required_bands)

    start_date, end_date = define_dates(
        start_year,
        start_month,
        start_day,
        duration_years,
        duration_months,
        duration_days,
    )
    if output_dir:
        export_path = get_output_path(
            grid_id=grid_id,
            start_date=start_date,
            end_date=end_date,
            sort_method=sort_method,
            mosaic_method=mosaic_method,
            required_bands=required_bands,
            output_dir=output_dir,
        )

    if output_dir:
        if export_path.exists() and not overwrite:
            return export_path

    items = search_for_items(
        bounds=bounds, grid_id=grid_id, start_date=start_date, end_date=end_date
    )

    if len(items) == 0:
        raise Exception(
            f"No scenes found for {grid_id} between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
        )

    items_with_orbits = add_item_info(items)

    sorted_items = sort_items(items=items_with_orbits, sort_method=sort_method)

    mosaic, profile = download_bands_pool(
        sorted_scenes=sorted_items,
        required_bands=required_bands,
        no_data_threshold=no_data_threshold,
        mosaic_method=mosaic_method,
        ocm_batch_size=ocm_batch_size,
        ocm_inference_dtype=ocm_inference_dtype,
    )
    if "visual" in required_bands:
        required_bands = ["Red", "Green", "Blue"]

    if output_dir:
        export_tif(mosaic, profile, export_path, required_bands)
        return export_path

    return mosaic, profile
