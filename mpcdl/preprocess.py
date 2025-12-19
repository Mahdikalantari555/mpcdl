"""
Preprocess module

This module provides functions for preprocessing and transforming geospatial data.

"""

import logging
from typing import Union, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import rioxarray
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)


def clip_raster(
    raster: Union[str, Path, xr.DataArray],
    geometry: Union[gpd.GeoDataFrame, Polygon],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> xr.DataArray:
    """
    Clip a raster dataset to a geometry.

    Args:
        raster: Path to raster file or xarray DataArray.
        geometry: GeoDataFrame or Polygon to clip to.
        output_path: Optional path to save clipped raster.
        **kwargs: Additional arguments for masking.

    Returns:
        Clipped xarray DataArray.
    """
    if isinstance(raster, (str, Path)):
        da = rioxarray.open_rasterio(raster)
    else:
        da = raster

    if isinstance(geometry, gpd.GeoDataFrame):
        geom = geometry.geometry.unary_union
    else:
        geom = geometry

    # Clip using rioxarray
    clipped = da.rio.clip([geom], **kwargs)

    if output_path:
        clipped.rio.to_raster(output_path)
        logger.info(f"Saved clipped raster to {output_path}")

    logger.info("Clipped raster to geometry")
    return clipped


def reproject_raster(
    raster: Union[str, Path, xr.DataArray],
    dst_crs: str,
    output_path: Optional[Union[str, Path]] = None,
    resolution: Optional[Tuple[float, float]] = None,
    resampling: Resampling = Resampling.nearest
) -> xr.DataArray:
    """
    Reproject a raster to a different coordinate reference system.

    Args:
        raster: Path to raster file or xarray DataArray.
        dst_crs: Target CRS (e.g., 'EPSG:4326').
        output_path: Optional path to save reprojected raster.
        resolution: Optional target resolution (x, y).
        resampling: Resampling method.

    Returns:
        Reprojected xarray DataArray.
    """
    if isinstance(raster, (str, Path)):
        da = rioxarray.open_rasterio(raster)
    else:
        da = raster

    # Reproject using rioxarray
    reprojected = da.rio.reproject(dst_crs, resolution=resolution, resampling=resampling)

    if output_path:
        reprojected.rio.to_raster(output_path)
        logger.info(f"Saved reprojected raster to {output_path}")

    logger.info(f"Reprojected raster to {dst_crs}")
    return reprojected


def resample_raster(
    raster: Union[str, Path, xr.DataArray],
    scale_factor: float = 1.0,
    resolution: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    resampling: Resampling = Resampling.nearest
) -> xr.DataArray:
    """
    Resample a raster to a different resolution.

    Args:
        raster: Path to raster file or xarray DataArray.
        scale_factor: Scale factor for resampling (e.g., 0.5 for half resolution).
        resolution: Target resolution (x, y) in units of the CRS.
        output_path: Optional path to save resampled raster.
        resampling: Resampling method.

    Returns:
        Resampled xarray DataArray.
    """
    if isinstance(raster, (str, Path)):
        da = rioxarray.open_rasterio(raster)
    else:
        da = raster

    if resolution:
        resampled = da.rio.reproject(da.rio.crs, resolution=resolution, resampling=resampling)
    else:
        # Use scale factor
        transform = da.rio.transform()
        new_transform = transform * transform.scale(scale_factor, scale_factor)
        resampled = da.rio.reproject(
            da.rio.crs,
            transform=new_transform,
            shape=tuple(int(s * scale_factor) for s in da.shape[-2:]),
            resampling=resampling
        )

    if output_path:
        resampled.rio.to_raster(output_path)
        logger.info(f"Saved resampled raster to {output_path}")

    logger.info(f"Resampled raster with scale factor {scale_factor}")
    return resampled


def normalize_raster(
    raster: Union[str, Path, xr.DataArray],
    method: str = "minmax",
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> xr.DataArray:
    """
    Normalize raster values.

    Args:
        raster: Path to raster file or xarray DataArray.
        method: Normalization method ('minmax', 'zscore', 'robust').
        output_path: Optional path to save normalized raster.
        **kwargs: Additional parameters for normalization.

    Returns:
        Normalized xarray DataArray.
    """
    if isinstance(raster, (str, Path)):
        da = rioxarray.open_rasterio(raster)
    else:
        da = raster

    data = da.values.astype(float)

    if method == "minmax":
        min_val = kwargs.get('min_val', np.nanmin(data))
        max_val = kwargs.get('max_val', np.nanmax(data))
        normalized = (data - min_val) / (max_val - min_val)
    elif method == "zscore":
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        normalized = (data - mean_val) / std_val
    elif method == "robust":
        median_val = np.nanmedian(data)
        mad_val = np.nanmedian(np.abs(data - median_val))
        normalized = (data - median_val) / mad_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized_da = da.copy()
    normalized_da.values = normalized

    if output_path:
        normalized_da.rio.to_raster(output_path)
        logger.info(f"Saved normalized raster to {output_path}")

    logger.info(f"Normalized raster using {method} method")
    return normalized_da


def stack_rasters(
    rasters: List[Union[str, Path, xr.DataArray]],
    output_path: Optional[Union[str, Path]] = None,
    band_names: Optional[List[str]] = None
) -> xr.DataArray:
    """
    Stack multiple rasters into a multi-band dataset.

    Args:
        rasters: List of raster paths or DataArrays.
        output_path: Optional path to save stacked raster.
        band_names: Optional names for each band.

    Returns:
        Stacked xarray DataArray.
    """
    data_arrays = []

    for raster in rasters:
        if isinstance(raster, (str, Path)):
            da = rioxarray.open_rasterio(raster)
        else:
            da = raster

        # Ensure single band
        if da.rio.count > 1:
            da = da.isel(band=0)

        data_arrays.append(da)

    # Stack along band dimension
    stacked = xr.concat(data_arrays, dim="band")

    if band_names:
        if len(band_names) != len(data_arrays):
            raise ValueError("band_names length must match number of rasters")
        stacked = stacked.assign_coords(band=band_names)

    if output_path:
        stacked.rio.to_raster(output_path)
        logger.info(f"Saved stacked raster to {output_path}")

    logger.info(f"Stacked {len(rasters)} rasters into multi-band dataset")
    return stacked


def preprocess_pipeline(
    raster: Union[str, Path, xr.DataArray],
    clip_geometry: Optional[Union[gpd.GeoDataFrame, Polygon]] = None,
    dst_crs: Optional[str] = None,
    resolution: Optional[Tuple[float, float]] = None,
    normalize_method: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> xr.DataArray:
    """
    Apply a preprocessing pipeline to a raster.

    Args:
        raster: Input raster.
        clip_geometry: Geometry to clip to.
        dst_crs: Target CRS for reprojection.
        resolution: Target resolution.
        normalize_method: Normalization method.
        output_path: Output path for final result.

    Returns:
        Processed xarray DataArray.
    """
    processed = raster

    if clip_geometry:
        processed = clip_raster(processed, clip_geometry)

    if dst_crs:
        processed = reproject_raster(processed, dst_crs, resolution=resolution)

    if normalize_method:
        processed = normalize_raster(processed, normalize_method)

    if output_path:
        processed.rio.to_raster(output_path)
        logger.info(f"Saved preprocessed raster to {output_path}")

    logger.info("Completed preprocessing pipeline")
    return processed