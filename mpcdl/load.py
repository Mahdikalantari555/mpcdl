"""
Load module

This module handles loading and reading geospatial data from various formats.

"""

import logging
from pathlib import Path
from typing import Union, Optional, Any, Dict
import rasterio
import rioxarray
import xarray as xr
import geopandas as gpd
from shapely.geometry import shape
import json

logger = logging.getLogger(__name__)


def load_raster(
    file_path: Union[str, Path],
    band: Optional[int] = None,
    **kwargs
) -> Union[rasterio.DatasetReader, xr.DataArray]:
    """
    Load a raster dataset from file.

    Args:
        file_path: Path to the raster file.
        band: Specific band to load (1-indexed). If None, loads all bands.
        **kwargs: Additional arguments passed to rasterio.open or rioxarray.open_rasterio.

    Returns:
        Rasterio DatasetReader if band is None, otherwise xarray DataArray.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Raster file not found: {file_path}")

    try:
        if band is not None:
            # Load specific band as DataArray
            da = rioxarray.open_rasterio(file_path, **kwargs)
            if band > da.rio.count:
                raise ValueError(f"Band {band} not available. Available bands: 1-{da.rio.count}")
            return da.isel(band=band-1)
        else:
            # Return DatasetReader for full access
            return rasterio.open(file_path, **kwargs)

    except Exception as e:
        logger.error(f"Failed to load raster {file_path}: {e}")
        raise


def load_vector(
    file_path: Union[str, Path],
    **kwargs
) -> gpd.GeoDataFrame:
    """
    Load a vector dataset from file.

    Args:
        file_path: Path to the vector file (Shapefile, GeoJSON, etc.).
        **kwargs: Additional arguments passed to geopandas.read_file.

    Returns:
        GeoDataFrame containing the vector data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Vector file not found: {file_path}")

    try:
        gdf = gpd.read_file(file_path, **kwargs)
        logger.info(f"Loaded vector data with {len(gdf)} features from {file_path}")
        return gdf
    except Exception as e:
        logger.error(f"Failed to load vector {file_path}: {e}")
        raise


def load_geojson_from_url(url: str, **kwargs) -> gpd.GeoDataFrame:
    """
    Load GeoJSON data directly from a URL.

    Args:
        url: URL to the GeoJSON file.
        **kwargs: Additional arguments passed to geopandas.GeoDataFrame.from_features.

    Returns:
        GeoDataFrame containing the GeoJSON data.
    """
    import requests

    try:
        response = requests.get(url)
        response.raise_for_status()
        geojson_data = response.json()

        gdf = gpd.GeoDataFrame.from_features(geojson_data['features'], **kwargs)
        logger.info(f"Loaded GeoJSON data with {len(gdf)} features from {url}")
        return gdf
    except Exception as e:
        logger.error(f"Failed to load GeoJSON from {url}: {e}")
        raise


def load_raster_from_url(
    url: str,
    band: Optional[int] = None,
    **kwargs
) -> Union[rasterio.DatasetReader, xr.DataArray]:
    """
    Load a raster dataset directly from a URL.

    Args:
        url: URL to the raster file.
        band: Specific band to load.
        **kwargs: Additional arguments.

    Returns:
        Raster data as DatasetReader or DataArray.
    """
    try:
        if band is not None:
            da = rioxarray.open_rasterio(url, **kwargs)
            return da.isel(band=band-1)
        else:
            # For URLs, we need to download to temp file first for rasterio
            import tempfile
            import requests

            response = requests.get(url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                return load_raster(tmp_file_path, band, **kwargs)
            finally:
                Path(tmp_file_path).unlink()

    except Exception as e:
        logger.error(f"Failed to load raster from {url}: {e}")
        raise


def load_stac_item_assets(
    item,
    asset_keys: Optional[list] = None,
    load_as: str = "dataarray"
) -> Dict[str, Any]:
    """
    Load assets from a STAC item.

    Args:
        item: STAC item object.
        asset_keys: List of asset keys to load. If None, loads all.
        load_as: "dataarray" for xarray DataArray, "dataset" for rasterio Dataset.

    Returns:
        Dictionary mapping asset keys to loaded data.
    """
    from .stac import MPCSTACClient

    client = MPCSTACClient()
    signed_urls = client.get_signed_urls(item)

    loaded_assets = {}
    keys_to_load = asset_keys if asset_keys else list(signed_urls.keys())

    for key in keys_to_load:
        if key not in signed_urls:
            logger.warning(f"Asset key '{key}' not found")
            continue

        url = signed_urls[key]
        try:
            if load_as == "dataarray":
                loaded_assets[key] = load_raster_from_url(url, band=1)
            elif load_as == "dataset":
                # This will require downloading to temp file
                import tempfile
                import requests

                response = requests.get(url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name

                loaded_assets[key] = load_raster(tmp_file_path)
                Path(tmp_file_path).unlink(missing_ok=True)
            else:
                raise ValueError(f"Invalid load_as value: {load_as}")

        except Exception as e:
            logger.error(f"Failed to load asset '{key}': {e}")

    logger.info(f"Loaded {len(loaded_assets)} assets from STAC item")
    return loaded_assets