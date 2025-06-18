import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Tuple
from numbagg import nanquantile
import numpy as np


def calculate_percentile_mosaic(
    all_scene_data: List[np.ndarray],
    valid_pixel_masks: List[np.ndarray],
    band_count: int,
    s2_scene_size: int,
    chunk_size: int = 100,
    max_workers: int = 8,
    percentile: float = 50.0,
) -> np.ndarray:
    """
    Memory-efficient percentile calculation processing row chunks in parallel.
    """
    logging.info("Calculating percentile mosaic using threaded row processing...")

    # Create row chunk specifications
    row_chunks = []
    for row_start in range(0, s2_scene_size, chunk_size):
        row_end = min(row_start + chunk_size, s2_scene_size)
        row_chunks.append((row_start, row_end))

    logging.info(f"Processing {len(row_chunks)} row chunks of {chunk_size} rows each")

    # Process row chunks in parallel
    process_chunk_partial = partial(
        process_row_chunk,
        all_scene_data=all_scene_data,
        valid_pixel_masks=valid_pixel_masks,
        band_count=band_count,
        percentile=percentile,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(
            executor.map(process_chunk_partial, row_chunks),
        )

    # Concatenate all row chunks back together
    mosaic = np.concatenate(chunk_results, axis=1)  # axis=1 is the height dimension

    logging.info("Threaded median mosaic calculation complete")
    return mosaic


def process_row_chunk(
    row_range: Tuple[int, int],
    all_scene_data: List[np.ndarray],
    valid_pixel_masks: List[np.ndarray],
    band_count: int,
    percentile: float,
) -> np.ndarray:
    """
    Process a chunk of rows to calculate percentile values.

    Args:
        row_range: (row_start, row_end)
        all_scene_data: List of scene arrays (bands, height, width)
        valid_pixel_masks: List of mask arrays (height, width)
        band_count: Number of bands
        scene_size: Full scene size (width)

    Returns:
        Percentile values for this row chunk (bands, chunk_height, scene_width)
    """
    row_start, row_end = row_range

    # Extract row chunk from all scenes - full width, specific rows
    chunk_data = np.stack(
        [scene[:, row_start:row_end, :] for scene in all_scene_data],
        axis=0,
    )  # (num_scenes, bands, chunk_height, scene_width)

    chunk_masks = np.stack(
        [mask[row_start:row_end, :] for mask in valid_pixel_masks],
        axis=0,
    )  # (num_scenes, chunk_height, scene_width)

    # Expand masks for all bands
    expanded_chunk_masks = np.expand_dims(chunk_masks, axis=1).repeat(
        band_count, axis=1
    )

    # Apply masks and calculate median
    masked_chunk = np.where(expanded_chunk_masks, chunk_data, np.nan)

    # If all values in a chunk are NaN, replace with 0.0
    all_nan_mask = np.all(np.isnan(masked_chunk), axis=0)
    masked_chunk = np.where(all_nan_mask, 0.0, masked_chunk)

    chunk_percentile = nanquantile(masked_chunk, percentile/100, axis=0)

    return chunk_percentile.astype(np.float32)