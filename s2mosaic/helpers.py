import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pystac
import pystac_client
import rasterio as rio
import scipy
import shapely
from dateutil.relativedelta import relativedelta
from omnicloudmask import predict_from_array
from pandas import DataFrame
from pystac.item_collection import ItemCollection
from tqdm.auto import tqdm

from .data_reader import get_band_with_mask, get_full_band
from .mosaic_methods import calculate_percentile_mosaic

SORT_VALID_DATA = "valid_data"
SORT_OLDEST = "oldest"
SORT_NEWEST = "newest"
SORT_CUSTOM = "custom"
MOSAIC_MEAN = "mean"
MOSAIC_FIRST = "first"
MOSAIC_PERCENTILE = "percentile"

VALID_SORT_METHODS = {SORT_VALID_DATA, SORT_OLDEST, SORT_NEWEST, SORT_CUSTOM}
VALID_MOSAIC_METHODS = {MOSAIC_MEAN, MOSAIC_FIRST, MOSAIC_PERCENTILE}


def download_bands_pool(
    sorted_scenes: pd.DataFrame,
    required_bands: List[str],
    coverage_mask: np.ndarray,
    no_data_threshold: Union[float, None],
    mosaic_method: str = "mean",
    ocm_batch_size: int = 6,
    ocm_inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
    percentile_value: float | None = 50.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:

    s2_scene_size = 10980
    possible_pixel_count = coverage_mask.sum()

    logging.info(f"Possible pixel count: {possible_pixel_count}")

    if "visual" in required_bands:
        band_count = 3
        band_indexes = [1, 2, 3]
        required_bands = required_bands * 3

    else:
        band_count = len(required_bands)
        band_indexes = [1] * len(required_bands)

    if mosaic_method == MOSAIC_PERCENTILE:
        # For percentile, we need to store all values for each pixel
        all_scene_data = []
        valid_pixel_masks = []
    else:
        # For mean and first, use the existing approach
        mosaic = np.zeros((band_count, s2_scene_size, s2_scene_size)).astype(np.float32)

    good_pixel_tracker = np.zeros((s2_scene_size, s2_scene_size)).astype(np.uint16)

    pbar = tqdm(
        total=len(sorted_scenes),
        desc=format_progress(0, len(sorted_scenes), 100.0),
        leave=False,
        bar_format="{desc}",
    )

    for index, item in enumerate(sorted_scenes["item"].tolist()):
        clear_pixels, good_pixels = ocm_cloud_mask(
            item=item,
            batch_size=ocm_batch_size,
            inference_dtype=ocm_inference_dtype,
            debug_cache=debug_cache,
            max_dl_workers=max_dl_workers,
        )
        combo_mask = (clear_pixels * good_pixels).astype(bool)

        # if method is first, only download valid, non cloudy pixels that have not been filled, else download all valid non cloudy pixels
        if mosaic_method == "first":
            combo_mask = (good_pixel_tracker == 0) & combo_mask
        
        
        good_pixel_tracker += combo_mask


        hrefs_and_indexes = [
            (item.assets[band].href, band_index)
            for band, band_index in zip(required_bands, band_indexes)
        ]

        get_band_with_mask_partial = partial(
            get_band_with_mask, mask=combo_mask, debug_cache=debug_cache,mosaic_method=mosaic_method
        )

        with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
            bands_and_profiles = list(
                executor.map(get_band_with_mask_partial, hrefs_and_indexes)
            )

        bands = []

        for band, profile in bands_and_profiles:
            bands.append(
                scipy.ndimage.zoom(
                    band,
                    (s2_scene_size / band.shape[0], s2_scene_size / band.shape[1]),
                    order=0,
                )
            )

        scene_data = np.array(bands)

        # Handle data accumulation based on method
        if mosaic_method == MOSAIC_PERCENTILE:
            # Store the scene data and mask for later percentile calculation
            all_scene_data.append(scene_data)
            valid_pixel_masks.append(combo_mask)
        else:
            # Existing logic for mean and first
            mosaic += scene_data

        completed_of_possible = coverage_mask * (good_pixel_tracker != 0)
        no_data_sum = coverage_mask.sum() - completed_of_possible.sum()
        logging.info(f"No data sum: {no_data_sum}")

        no_data_pct = (1 - (completed_of_possible.sum() / possible_pixel_count)) * 100
        logging.info(f"No data pct: {no_data_pct}")

        pbar.set_description(
            format_progress(index + 1, len(sorted_scenes), no_data_pct)
        )

        if mosaic_method == "first":
            if no_data_sum == 0:
                break
        # if no_data_threshold is set, stop if threshold is reached
        if no_data_threshold is not None:
            if no_data_sum < (possible_pixel_count * no_data_threshold):
                break
        pbar.update(1)

    remaining_scenes = pbar.total - pbar.n
    pbar.update(remaining_scenes)
    pbar.refresh()
    pbar.close()

    if mosaic_method in ["percentile"]:
        if percentile_value is None:
            raise ValueError("Percentile must be provided for percentile mosaic method")

        max_workers = multiprocessing.cpu_count() // 2

        mosaic = calculate_percentile_mosaic(
            all_scene_data=all_scene_data,
            valid_pixel_masks=valid_pixel_masks,
            band_count=band_count,
            s2_scene_size=s2_scene_size,
            max_workers=max_workers,
            percentile_value=float(percentile_value),
        )

    if mosaic_method == "mean":
        mosaic = np.divide(
            mosaic,
            good_pixel_tracker,
            out=np.zeros_like(mosaic),
            where=good_pixel_tracker != 0,
        )

    mosaic = normalise_for_output(
        mosaic=mosaic,
        required_bands=required_bands,
    )

    return mosaic, profile


def get_valid_mask(bands: np.ndarray, dilation_count: int = 4) -> np.ndarray:
    # create mask to remove pixels with no data, add dilation to remove edge pixels
    no_data = (bands.sum(axis=0) == 0).astype(np.uint8)
    # erode mask to remove edge pixels
    if dilation_count > 0:
        no_data = scipy.ndimage.binary_dilation(no_data, iterations=dilation_count)
    return ~no_data


def ocm_cloud_mask(
    item: pystac.Item,
    batch_size: int = 6,
    inference_dtype: str = "bf16",
    debug_cache: bool = False,
    max_dl_workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    # download RG+NIR bands at 20m resolution for cloud masking
    required_bands = ["B04", "B03", "B8A"]
    get_band_20m = partial(get_full_band, res=20, debug_cache=debug_cache)

    hrefs = [item.assets[band].href for band in required_bands]

    with ThreadPoolExecutor(max_workers=max_dl_workers) as executor:
        bands_and_profiles = list(executor.map(get_band_20m, hrefs))

    # Separate bands and profiles
    bands, _ = zip(*bands_and_profiles)
    ocm_input = np.vstack(bands)

    # no_data_mask = get_no_data_mask()
    mask = (
        predict_from_array(
            input_array=ocm_input,
            batch_size=batch_size,
            inference_dtype=inference_dtype,
        )[0]
        == 0
    )
    # interpolate mask back to 10m
    mask = mask.repeat(2, axis=0).repeat(2, axis=1)
    valid_mask = get_valid_mask(ocm_input)
    valid_mask = valid_mask.repeat(2, axis=0).repeat(2, axis=1)
    return mask, valid_mask


def format_progress(current, total, no_data_pct):
    return f"Scenes: {current}/{total} | Mosaic currently contains {no_data_pct:.2f}% no data pixels"


def normalise_for_output(
    mosaic: np.ndarray,
    required_bands: List[str],
):
    if "visual" in required_bands:
        mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    else:
        mosaic = np.clip(mosaic, 0, 65535).astype(np.int16)
    return mosaic


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
    nodata_value: Union[int, None] = 0,
) -> None:
    profile.update(
        count=array.shape[0], dtype=array.dtype, nodata=nodata_value, compress="lzw"
    )
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)
        dst.descriptions = required_bands


def search_for_items(
    bounds,
    grid_id: str,
    start_date: date,
    end_date: date,
    additional_query: Dict[str, Any],
) -> ItemCollection:

    base_query = {"s2:mgrs_tile": {"eq": grid_id}}
    if additional_query:
        base_query.update(additional_query)

    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        "query": base_query,
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
        Path(__file__).resolve().parent / "S2_grid/sentinel_2_index.gpkg"
    )

    # Use SQLite query to filter by Name directly
    query = f"SELECT * FROM sentinel_2_index WHERE Name = '{grid_id}'"

    try:
        # Read only the matching row directly from the gpkg
        grid_entry = gpd.read_file(S2_grid_file, sql=query)

        if len(grid_entry) != 1:
            raise ValueError(
                f"Grid {grid_id} not found. It should be in the format '50HMH'. "
                "For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"
            )
        return grid_entry.iloc[0].geometry

    except Exception as e:
        logging.error(f"Error reading grid entry: {e}")
        raise


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


def validate_inputs(
    sort_method: str,
    mosaic_method: str,
    no_data_threshold: Union[float, None],
    required_bands: List[str],
    grid_id: str,
    percentile_value: Optional[float],
) -> None:
    if not grid_id.isalnum() or not grid_id.isupper():
        raise ValueError(
            f"Grid {grid_id} is invalid. It should be in the format '50HMH'. "
            "For more info on the S2 grid system visit https://sentiwiki.copernicus.eu/web/s2-products"
        )
    if sort_method not in VALID_SORT_METHODS:
        raise ValueError(
            f"Invalid sort method: {sort_method}. Must be one of {VALID_SORT_METHODS}"
        )
    if mosaic_method not in VALID_MOSAIC_METHODS:
        if mosaic_method == "median":
            logging.warning(
                "Median mosaic method is deprecated, use percentile with 50th percentile instead"
            )
            mosaic_method = MOSAIC_PERCENTILE
            percentile_value = 50.0
        else:
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

    if mosaic_method != MOSAIC_PERCENTILE:
        if percentile_value is not None:
            raise ValueError(
                f"percentile_value is only valid for percentile mosaic method, got {percentile_value}"
            )

    if mosaic_method == MOSAIC_PERCENTILE:
        if percentile_value is None:
            raise ValueError("percentile_value must be provided for percentile mosaic method")
        if percentile_value < 0 or percentile_value > 100:
            raise ValueError(f"percentile_value must be between 0 and 100, got {percentile_value}")


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

