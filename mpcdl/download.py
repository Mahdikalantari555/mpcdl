"""
Download module

This module handles downloading data from STAC catalogs and other sources.

"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from planetary_computer import sign
import geopandas as gpd
import stackstac
import rioxarray
from shapely.geometry import box
import pystac

logger = logging.getLogger(__name__)


def _get_utm_epsg_from_bbox(bbox: List[float]) -> int:
    """
    Calculate UTM EPSG code from a bounding box.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat].

    Returns:
        UTM EPSG code.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    zone = int((center_lon + 180) // 6) + 1
    if center_lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone


def download_file(
    url: str,
    output_path: Union[str, Path],
    sign_url: bool = True,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Download a file from a URL to a local path.

    Args:
        url: The URL to download from.
        output_path: Local path to save the file.
        sign_url: Whether to sign the URL for MPC access.
        overwrite: Whether to overwrite existing files.
        progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes).

    Returns:
        True if download successful, False otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        logger.info(f"File already exists at {output_path}, skipping download")
        return True

    try:
        if sign_url and url.startswith("https://"):
            url = sign(url)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            if progress_callback:
                                progress_callback(len(chunk), total_size)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logger.info(f"Downloaded file to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_stac_assets(
    item_assets: Dict[str, str],
    output_dir: Union[str, Path],
    asset_keys: Optional[List[str]] = None,
    sign_urls: bool = True,
    overwrite: bool = False
) -> Dict[str, Path]:
    """
    Download multiple assets from a STAC item.

    Args:
        item_assets: Dictionary of asset key to URL.
        output_dir: Directory to save assets.
        asset_keys: Specific asset keys to download. If None, download all.
        sign_urls: Whether to sign URLs.
        overwrite: Whether to overwrite existing files.

    Returns:
        Dictionary mapping asset keys to local file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = {}
    assets_to_download = asset_keys if asset_keys else list(item_assets.keys())

    for asset_key in assets_to_download:
        if asset_key not in item_assets:
            logger.warning(f"Asset key '{asset_key}' not found in item assets")
            continue

        url = item_assets[asset_key]
        filename = Path(urlparse(url).path).name
        output_path = output_dir / filename

        if download_file(url, output_path, sign_urls, overwrite):
            downloaded_files[asset_key] = output_path
        else:
            logger.error(f"Failed to download asset '{asset_key}'")

    logger.info(f"Downloaded {len(downloaded_files)} assets to {output_dir}")
    return downloaded_files


def download_batch(
    urls: List[str],
    output_dir: Union[str, Path],
    filenames: Optional[List[str]] = None,
    sign_urls: bool = True,
    overwrite: bool = False,
    max_workers: int = 4
) -> List[Path]:
    """
    Download multiple files in parallel.

    Args:
        urls: List of URLs to download.
        output_dir: Directory to save files.
        filenames: Optional list of filenames. If None, use URL basename.
        sign_urls: Whether to sign URLs.
        overwrite: Whether to overwrite existing files.
        max_workers: Maximum number of parallel downloads.

    Returns:
        List of downloaded file paths.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filenames and len(filenames) != len(urls):
        raise ValueError("filenames list must match urls list length")

    downloaded_files = []

    def download_single(url: str, filename: Optional[str] = None) -> Optional[Path]:
        if filename:
            output_path = output_dir / filename
        else:
            output_path = output_dir / Path(urlparse(url).path).name

        if download_file(url, output_path, sign_urls, overwrite):
            return output_path
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single, url, filenames[i] if filenames else None)
                  for i, url in enumerate(urls)]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading batch"):
            result = future.result()
            if result:
                downloaded_files.append(result)

    logger.info(f"Downloaded {len(downloaded_files)} files in batch")
    return downloaded_files


def download_mpc_dataset(
    collection: str,
    item_id: str,
    output_dir: Union[str, Path],
    asset_keys: Optional[List[str]] = None
) -> Optional[Dict[str, Path]]:
    """
    Download assets from an MPC dataset item.

    Args:
        collection: Collection ID.
        item_id: Item ID.
        output_dir: Output directory.
        asset_keys: Specific assets to download.

    Returns:
        Dictionary of downloaded asset paths, or None if failed.
    """
    from .stac import MPCSTACClient

    client = MPCSTACClient()
    item = client.get_item(item_id, collection)

    if not item:
        logger.error(f"Could not find item '{item_id}' in collection '{collection}'")
        return None

    signed_urls = client.get_signed_urls(item)
    return download_stac_assets(signed_urls, output_dir, asset_keys)


def download_stac_assets_clipped(
    stac_items: Union[List, object],  # pystac.Item or List[pystac.Item]
    asset_keys: List[str],
    bbox: List[float],
    output_dir: Union[str, Path],
    resolution: float = 10.0,
    sign_items: bool = True,
    overwrite: bool = False,
    show_progress: bool = True,
    stack_bands: bool = False,
    stacked_filename: Optional[str] = None
) -> Dict[str, Path]:
    """
    Download and clip STAC assets using stackstac for efficient processing.

    Args:
        stac_items: Single STAC item or list of STAC items.
        asset_keys: List of asset keys to download and clip.
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat].
        output_dir: Directory to save the clipped GeoTIFF files.
        resolution: Spatial resolution in meters.
        sign_items: Whether to sign item URLs for MPC access.
        overwrite: Whether to overwrite existing files.
        show_progress: Whether to show progress bar during computation.
        stack_bands: Whether to create a single multi-band TIFF with all assets.
        stacked_filename: Filename for the stacked multi-band TIFF (if stack_bands=True).

    Returns:
        Dictionary mapping asset keys to output file paths. If stack_bands=True,
        also includes 'stacked' key with the multi-band file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure stac_items is a list
    if not isinstance(stac_items, list):
        stac_items = [stac_items]

    # Sign items if needed
    if sign_items:
        signed_items = []
        for item in stac_items:
            signed_item = item.clone()
            for asset_key, asset in signed_item.assets.items():
                if asset.href.startswith("https://"):
                    asset.href = sign(asset.href)
            signed_items.append(signed_item)
        stac_items = signed_items

    # Calculate UTM EPSG
    utm_epsg = _get_utm_epsg_from_bbox(bbox)

    # Get bounds in lat/lon for lazy cropping
    bounds_latlon = tuple(bbox)

    # Stack the items
    stack = stackstac.stack(
        stac_items,
        assets=asset_keys,
        resolution=resolution,
        epsg=utm_epsg,
        bounds_latlon=bounds_latlon
    )

    # Compute the stack with optional progress bar
    if show_progress:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            stack = stack.compute()
    else:
        stack = stack.compute()

    # Save each band as separate GeoTIFF
    downloaded_files = {}
    for band in stack.band.values:
        band_data = stack.sel(band=band)
        output_path = output_dir / f"{band}.tif"
        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists, skipping")
            downloaded_files[str(band)] = output_path
            continue
        band_data.rio.to_raster(str(output_path))
        logger.info(f"Saved clipped asset '{band}' to {output_path}")
        downloaded_files[str(band)] = output_path

    # Optionally create stacked multi-band TIFF
    if stack_bands:
        if stacked_filename is None:
            # Use item ID if available, otherwise generic name
            if hasattr(stac_items[0], 'id'):
                stacked_filename = f"{stac_items[0].id}_stacked.tif"
            else:
                stacked_filename = "stacked_bands.tif"
        
        stacked_path = output_dir / stacked_filename
        if stacked_path.exists() and not overwrite:
            logger.info(f"Stacked file {stacked_path} already exists, skipping")
        else:
            # Prepare the stack for rioxarray - ensure it's (band, y, x)
            # Remove time dimension if it exists (should be 1)
            if 'time' in stack.dims:
                stack_for_save = stack.isel(time=0)
            else:
                stack_for_save = stack
            
            # Ensure band dimension is first
            if stack_for_save.dims[0] != 'band':
                stack_for_save = stack_for_save.transpose('band', ...)
            
            stack_for_save.rio.to_raster(str(stacked_path))
            logger.info(f"Saved stacked multi-band TIFF to {stacked_path}")
        
        downloaded_files['stacked'] = stacked_path


    logger.info(f"Downloaded and clipped {len(downloaded_files)} assets to {output_dir}")
    return downloaded_files
