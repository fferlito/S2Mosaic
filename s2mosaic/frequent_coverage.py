import logging
from typing import List

import geopandas as gpd
import numpy as np
import scipy
from geopandas import GeoDataFrame
from pystac.item import Item
from pystac.item_collection import ItemCollection
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon


def get_coverage(scenes: List[Item]) -> gpd.GeoDataFrame:
    extents = []
    for scene in scenes:
        if scene.geometry is not None and "coordinates" in scene.geometry:
            extents.append(Polygon(scene.geometry["coordinates"][0]))

    extent_gdf = gpd.GeoDataFrame(geometry=extents, crs="EPSG:4326")  # type: ignore
    return extent_gdf


def get_raster_coverage(
    scene_bounds: Polygon, coverage_gdf: GeoDataFrame, local_crs: int, resolution=10
):
    scene_gdf = gpd.GeoDataFrame(
        [scene_bounds], geometry=[scene_bounds], crs="EPSG:4326"
    ).to_crs(
        f"EPSG:{local_crs}"
    )  # type: ignore
    coverage_gdf_local = coverage_gdf.to_crs(f"EPSG:{local_crs}")

    coverage_gdf_local["geometry"] = coverage_gdf_local.buffer(0)  # type: ignore

    extent = scene_gdf.total_bounds  # type: ignore
    x_min, _, _, y_max = extent

    raster = np.zeros((10980, 10980), dtype=np.int16)

    transform = Affine(resolution, 0, x_min, 0, -resolution, y_max)

    for _, coverage_row in coverage_gdf_local.iterrows():  # type: ignore
        geom = coverage_row.geometry
        raster += rasterize(
            [(geom, 1)],
            out_shape=raster.shape,
            fill=0,
            dtype=np.int16,
            transform=transform,
        )
    return raster


def get_frequent_coverage(
    scene_bounds: Polygon, scenes: ItemCollection, coverage_threshold_pct=0.1
) -> np.ndarray:
    scenes_list = list(scenes)
    logging.info(f"Calculating total coverage for {len(scenes_list)} scenes")

    try:
        local_crs = scenes_list[0].properties["proj:epsg"]
    except KeyError:
        local_crs = scenes_list[0].properties["proj:code"]
        local_crs = int(local_crs.split(":")[-1])

    logging.info(f"Using local CRS: EPSG:{local_crs}")

    coverage_gdf = get_coverage(scenes_list)
    raster = get_raster_coverage(scene_bounds, coverage_gdf, local_crs)
    logging.info(f"Coverage raster shape: {raster.shape}")

    max_count = raster.max()
    logging.info(f"Max coverage count: {max_count}")
    # any area that is covered by more than 10% of the scenes is considered to be covered
    dymanic_threshold = max_count * coverage_threshold_pct
    logging.info(f"Dynamic threshold: {dymanic_threshold}")
    # threshold the raster to get a mask of the frequent data
    frequent_data_mask = raster >= dymanic_threshold
    # expand the mask to include nearby pixels, this grows the no data areas by 4 pixels
    frequent_data_mask = ~scipy.ndimage.binary_dilation(
        ~frequent_data_mask, iterations=4
    )
    return frequent_data_mask
