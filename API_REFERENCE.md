# mpcdl API Reference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive reference for the mpcdl library, which provides tools for downloading and processing STAC data from Microsoft Planetary Computer.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [STAC Module](#stac-module)
  - [MPCSTACClient Class](#mpcstacclient-class)
  - [Functions](#stac-functions)
- [Download Module](#download-module)
- [Load Module](#load-module)
- [Preprocess Module](#preprocess-module)
- [Utils Module](#utils-module)
- [Examples](#examples)

## Overview

mpcdl is a Python library that simplifies access to geospatial data from Microsoft Planetary Computer (MPC). It provides high-level APIs for:

- Searching STAC catalogs
- Downloading data with automatic URL signing
- Loading various geospatial formats
- Preprocessing raster data
- Utility functions for configuration and logging

The library is organized into several modules, each focusing on a specific aspect of the data pipeline.

## Installation

```bash
pip install git+https://github.com/Mahdikalantari555/mpcdl.git
```

For full functionality with geospatial processing:

```bash
pip install mpcdl[geo]
```

## STAC Module

The STAC module provides utilities for working with STAC (SpatioTemporal Asset Catalog) data from Microsoft Planetary Computer.

### MPCSTACClient Class

A client for interacting with Microsoft Planetary Computer STAC API.

#### `__init__(api_url=None)`

Initialize the MPC STAC client.

**Parameters:**
- `api_url` (Optional[str]): Custom STAC API URL. Defaults to MPC STAC URL.

**Example:**
```python
from mpcdl import MPCSTACClient

client = MPCSTACClient()
```

#### `search_items(collection, bbox=None, datetime_range=None, limit=100, **kwargs)`

Search for STAC items in a collection.

**Parameters:**
- `collection` (str): Collection ID to search in
- `bbox` (Optional[List[float]]): Bounding box as [min_lon, min_lat, max_lon, max_lat]
- `datetime_range` (Optional[str]): Date range in ISO 8601 format (e.g., "2020-01-01/2020-12-31")
- `limit` (int): Maximum number of items to return
- `**kwargs`: Additional search parameters

**Returns:** List of matching STAC items

**Example:**
```python
items = client.search_items(
    collection="sentinel-2-l2a",
    bbox=[-122.5, 37.7, -122.3, 37.8],
    datetime_range="2023-01-01/2023-12-31",
    limit=10
)
```

#### `get_item(item_id, collection)`

Retrieve a specific STAC item by ID.

**Parameters:**
- `item_id` (str): The item ID
- `collection` (str): The collection containing the item

**Returns:** The STAC item if found, None otherwise

#### `get_signed_urls(item)`

Get signed URLs for all assets in a STAC item.

**Parameters:**
- `item` (pystac.Item): The STAC item containing assets

**Returns:** Dictionary mapping asset keys to signed URLs

#### `items_to_geodataframe(items)`

Convert STAC items to a GeoDataFrame.

**Parameters:**
- `items` (List[pystac.Item]): List of STAC items

**Returns:** GeoDataFrame with item properties and geometries

### STAC Functions

#### `search_mpc_collection(collection, bbox=None, datetime_range=None, limit=100, path=None, row=None, mgrs_tile=None, require_full_bbox_coverage=False)`

Convenience function to search items in an MPC collection.

**Parameters:**
- `collection` (str): Collection ID
- `bbox` (Optional[List[float]]): Bounding box
- `datetime_range` (Optional[str]): Date range
- `limit` (int): Max items
- `path` (Optional[int]): WRS path number (for Landsat collections)
- `row` (Optional[int]): WRS row number (for Landsat collections)
- `mgrs_tile` (Optional[str]): MGRS tile ID (for Sentinel-2 collections)
- `require_full_bbox_coverage` (bool): If True, only return items whose footprint fully covers the bbox

**Returns:** List of STAC items

#### `get_mpc_item(item_id, collection)`

Convenience function to get a specific MPC item.

**Parameters:**
- `item_id` (str): Item ID
- `collection` (str): Collection ID

**Returns:** STAC item if found

## Download Module

The download module handles downloading data from STAC catalogs and other sources.

### `download_file(url, output_path, sign_url=True, overwrite=False, progress_callback=None)`

Download a file from a URL to a local path.

**Parameters:**
- `url` (str): The URL to download from
- `output_path` (Union[str, Path]): Local path to save the file
- `sign_url` (bool): Whether to sign the URL for MPC access
- `overwrite` (bool): Whether to overwrite existing files
- `progress_callback` (Optional[Callable]): Optional callback for progress updates

**Returns:** True if download successful, False otherwise

### `download_stac_assets(item_assets, output_dir, asset_keys=None, sign_urls=True, overwrite=False)`

Download multiple assets from a STAC item.

**Parameters:**
- `item_assets` (Dict[str, str]): Dictionary of asset key to URL
- `output_dir` (Union[str, Path]): Directory to save assets
- `asset_keys` (Optional[List[str]]): Specific asset keys to download
- `sign_urls` (bool): Whether to sign URLs
- `overwrite` (bool): Whether to overwrite existing files

**Returns:** Dictionary mapping asset keys to local file paths

### `download_batch(urls, output_dir, filenames=None, sign_urls=True, overwrite=False, max_workers=4)`

Download multiple files in parallel.

**Parameters:**
- `urls` (List[str]): List of URLs to download
- `output_dir` (Union[str, Path]): Directory to save files
- `filenames` (Optional[List[str]]): Optional list of filenames
- `sign_urls` (bool): Whether to sign URLs
- `overwrite` (bool): Whether to overwrite existing files
- `max_workers` (int): Maximum number of parallel downloads

**Returns:** List of downloaded file paths

### `download_mpc_dataset(collection, item_id, output_dir, asset_keys=None)`

Download assets from an MPC dataset item.

**Parameters:**
- `collection` (str): Collection ID
- `item_id` (str): Item ID
- `output_dir` (Union[str, Path]): Output directory
- `asset_keys` (Optional[List[str]]): Specific assets to download

**Returns:** Dictionary of downloaded asset paths, or None if failed

### `download_stac_assets_clipped(stac_items, asset_keys, bbox, output_dir, resolution=10.0, sign_items=True, overwrite=False, show_progress=True, stack_bands=False, stacked_filename=None)`

Download and clip STAC assets using stackstac for efficient processing.

**Parameters:**
- `stac_items` (Union[List, object]): Single STAC item or list of STAC items
- `asset_keys` (List[str]): List of asset keys to download and clip
- `bbox` (List[float]): Bounding box as [min_lon, min_lat, max_lon, max_lat]
- `output_dir` (Union[str, Path]): Directory to save the clipped GeoTIFF files
- `resolution` (float): Spatial resolution in meters
- `sign_items` (bool): Whether to sign item URLs for MPC access
- `overwrite` (bool): Whether to overwrite existing files
- `show_progress` (bool): Whether to show progress bar during computation
- `stack_bands` (bool): Whether to create a single multi-band TIFF with all assets
- `stacked_filename` (Optional[str]): Filename for the stacked multi-band TIFF

**Returns:** Dictionary mapping asset keys to output file paths

## Load Module

The load module handles loading and reading geospatial data from various formats.

### `load_raster(file_path, band=None, **kwargs)`

Load a raster dataset from file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the raster file
- `band` (Optional[int]): Specific band to load (1-indexed)
- `**kwargs`: Additional arguments passed to rasterio.open or rioxarray.open_rasterio

**Returns:** Rasterio DatasetReader if band is None, otherwise xarray DataArray

### `load_vector(file_path, **kwargs)`

Load a vector dataset from file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the vector file
- `**kwargs`: Additional arguments passed to geopandas.read_file

**Returns:** GeoDataFrame containing the vector data

### `load_geojson_from_url(url, **kwargs)`

Load GeoJSON data directly from a URL.

**Parameters:**
- `url` (str): URL to the GeoJSON file
- `**kwargs`: Additional arguments passed to geopandas.GeoDataFrame.from_features

**Returns:** GeoDataFrame containing the GeoJSON data

### `load_raster_from_url(url, band=None, **kwargs)`

Load a raster dataset directly from a URL.

**Parameters:**
- `url` (str): URL to the raster file
- `band` (Optional[int]): Specific band to load
- `**kwargs`: Additional arguments

**Returns:** Raster data as DatasetReader or DataArray

### `load_stac_item_assets(item, asset_keys=None, load_as="dataarray")`

Load assets from a STAC item.

**Parameters:**
- `item`: STAC item object
- `asset_keys` (Optional[list]): List of asset keys to load
- `load_as` (str): "dataarray" for xarray DataArray, "dataset" for rasterio Dataset

**Returns:** Dictionary mapping asset keys to loaded data

## Preprocess Module

The preprocess module provides functions for preprocessing and transforming geospatial data.

### `clip_raster(raster, geometry, output_path=None, **kwargs)`

Clip a raster dataset to a geometry.

**Parameters:**
- `raster` (Union[str, Path, xr.DataArray]): Path to raster file or xarray DataArray
- `geometry` (Union[gpd.GeoDataFrame, Polygon]): GeoDataFrame or Polygon to clip to
- `output_path` (Optional[Union[str, Path]]): Optional path to save clipped raster
- `**kwargs`: Additional arguments for masking

**Returns:** Clipped xarray DataArray

### `reproject_raster(raster, dst_crs, output_path=None, resolution=None, resampling=Resampling.nearest)`

Reproject a raster to a different coordinate reference system.

**Parameters:**
- `raster` (Union[str, Path, xr.DataArray]): Path to raster file or xarray DataArray
- `dst_crs` (str): Target CRS (e.g., 'EPSG:4326')
- `output_path` (Optional[Union[str, Path]]): Optional path to save reprojected raster
- `resolution` (Optional[Tuple[float, float]]): Optional target resolution (x, y)
- `resampling` (Resampling): Resampling method

**Returns:** Reprojected xarray DataArray

### `resample_raster(raster, scale_factor=1.0, resolution=None, output_path=None, resampling=Resampling.nearest)`

Resample a raster to a different resolution.

**Parameters:**
- `raster` (Union[str, Path, xr.DataArray]): Path to raster file or xarray DataArray
- `scale_factor` (float): Scale factor for resampling
- `resolution` (Optional[Tuple[float, float]]): Target resolution (x, y) in units of the CRS
- `output_path` (Optional[Union[str, Path]]): Optional path to save resampled raster
- `resampling` (Resampling): Resampling method

**Returns:** Resampled xarray DataArray

### `normalize_raster(raster, method="minmax", output_path=None, **kwargs)`

Normalize raster values.

**Parameters:**
- `raster` (Union[str, Path, xr.DataArray]): Path to raster file or xarray DataArray
- `method` (str): Normalization method ('minmax', 'zscore', 'robust')
- `output_path` (Optional[Union[str, Path]]): Optional path to save normalized raster
- `**kwargs`: Additional parameters for normalization

**Returns:** Normalized xarray DataArray

### `stack_rasters(rasters, output_path=None, band_names=None)`

Stack multiple rasters into a multi-band dataset.

**Parameters:**
- `rasters` (List[Union[str, Path, xr.DataArray]]): List of raster paths or DataArrays
- `output_path` (Optional[Union[str, Path]]): Optional path to save stacked raster
- `band_names` (Optional[List[str]]): Optional names for each band

**Returns:** Stacked xarray DataArray

### `preprocess_pipeline(raster, clip_geometry=None, dst_crs=None, resolution=None, normalize_method=None, output_path=None)`

Apply a preprocessing pipeline to a raster.

**Parameters:**
- `raster` (Union[str, Path, xr.DataArray]): Input raster
- `clip_geometry` (Optional[Union[gpd.GeoDataFrame, Polygon]]): Geometry to clip to
- `dst_crs` (Optional[str]): Target CRS for reprojection
- `resolution` (Optional[Tuple[float, float]]): Target resolution
- `normalize_method` (Optional[str]): Normalization method
- `output_path` (Optional[Union[str, Path]]): Output path for final result

**Returns:** Processed xarray DataArray

## Utils Module

The utils module contains utility functions and helpers.

### `setup_logging(level="INFO", format_string=None, log_file=None)`

Setup logging configuration for the package.

**Parameters:**
- `level` (Union[str, int]): Logging level
- `format_string` (Optional[str]): Custom format string for log messages
- `log_file` (Optional[Union[str, Path]]): Optional file path to save logs

### `load_environment_variables(env_file=".env")`

Load environment variables from a .env file.

**Parameters:**
- `env_file` (Union[str, Path]): Path to the .env file

### `get_env_var(key, default=None)`

Get an environment variable with optional default.

**Parameters:**
- `key` (str): Environment variable key
- `default` (Optional[str]): Default value if key not found

**Returns:** Environment variable value or default

### `ensure_directory(path)`

Ensure a directory exists, creating it if necessary.

**Parameters:**
- `path` (Union[str, Path]): Directory path

**Returns:** Path object for the directory

### `log_package_info()`

Log package version and dependency information.

### `read_yaml_config(config_path)`

Read configuration from a YAML file.

**Parameters:**
- `config_path` (Union[str, Path]): Path to the YAML configuration file

**Returns:** Dictionary containing configuration data

### `write_yaml_config(config, config_path)`

Write configuration to a YAML file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `config_path` (Union[str, Path]): Path to save the YAML file

### `read_json_config(config_path)`

Read configuration from a JSON file.

**Parameters:**
- `config_path` (Union[str, Path]): Path to the JSON configuration file

**Returns:** Dictionary containing configuration data

### `write_json_config(config, config_path)`

Write configuration to a JSON file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `config_path` (Union[str, Path]): Path to save the JSON file

### `get_file_size_mb(file_path)`

Get the size of a file in megabytes.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the file

**Returns:** File size in MB

### `validate_bbox(bbox)`

Validate a bounding box.

**Parameters:**
- `bbox` (list): Bounding box as [min_lon, min_lat, max_lon, max_lat]

**Returns:** True if valid, False otherwise

### `timestamp_filename(prefix="", suffix="", extension="")`

Generate a timestamped filename.

**Parameters:**
- `prefix` (str): Prefix for the filename
- `suffix` (str): Suffix for the filename
- `extension` (str): File extension (without dot)

**Returns:** Timestamped filename string

## Examples

### Basic Search and Download

```python
import mpcdl

# Search for Sentinel-2 data
items = mpcdl.search_mpc_collection(
    collection="sentinel-2-l2a",
    bbox=[-122.5, 37.7, -122.3, 37.8],
    datetime_range="2023-06-01/2023-06-30",
    limit=5
)

# Download RGB bands from first item
if items:
    assets = mpcdl.download_stac_assets(
        items[0].assets,
        output_dir="./data",
        asset_keys=["B04", "B03", "B02"]
    )
```

### Load and Preprocess Data

```python
import mpcdl

# Load raster data
raster = mpcdl.load_raster("./data/B04.tif")

# Clip to area of interest
aoi = mpcdl.load_vector("./aoi.geojson")
clipped = mpcdl.clip_raster(raster, aoi)

# Reproject and resample
reprojected = mpcdl.reproject_raster(clipped, "EPSG:4326")
resampled = mpcdl.resample_raster(reprojected, resolution=(0.0001, 0.0001))

# Save result
resampled.rio.to_raster("./processed_data.tif")
```

### Advanced Download with Clipping

```python
import mpcdl

# Search for items
items = mpcdl.search_mpc_collection(
    collection="sentinel-2-l2a",
    bbox=[-122.5, 37.7, -122.3, 37.8],
    datetime_range="2023-06-01/2023-06-30",
    limit=3
)

# Download and clip multiple bands directly
downloaded = mpcdl.download_stac_assets_clipped(
    stac_items=items,
    asset_keys=["B04", "B03", "B02", "B08"],
    bbox=[-122.5, 37.7, -122.3, 37.8],
    output_dir="./clipped_data",
    resolution=10.0,
    stack_bands=True,
    stacked_filename="sentinel_rgb_nir.tif"
)
```

---

For more examples and tutorials, visit the [GitHub repository](https://github.com/Mahdikalantari555/mpcdl).