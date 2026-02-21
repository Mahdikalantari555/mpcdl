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
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import socket
from tqdm import tqdm
from planetary_computer import sign
import geopandas as gpd
import stackstac
import rioxarray
from shapely.geometry import box
import pystac
import dask
import psutil

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename to prevent path traversal attacks and ensure valid filenames."""
    import re
    
    logger.debug(f"DEBUG: Sanitizing filename: '{filename}' (length: {len(filename)})")
    
    # Remove path separators
    filename = filename.replace('/', '').replace('\\', '')
    # Remove .. sequences
    filename = re.sub(r'\.\.+', '', filename)
    
    # Keep only safe characters: alphanumeric, dot, hyphen, underscore
    filename = re.sub(r'[^\w\.-]', '_', filename)
    
    # Remove multiple consecutive underscores/dots/hyphens
    filename = re.sub(r'[_-]{2,}', '_', filename)
    filename = re.sub(r'\.{2,}', '.', filename)
    
    # Remove leading/trailing special characters
    filename = filename.strip('_.-')
    
    # Limit length - use shorter limit to avoid filesystem issues
    if len(filename) > max_length:
        # Keep extension if present
        if '.' in filename:
            name_part, ext = filename.rsplit('.', 1)
            filename = name_part[:max_length-len(ext)-1] + '.' + ext
        else:
            filename = filename[:max_length]
    
    # Ensure not empty
    if not filename:
        filename = 'default_filename'
    
    logger.debug(f"DEBUG: Sanitized filename: '{filename}' (length: {len(filename)})")
    return filename


def validate_and_download_url(url: str, retries: int = 3, backoff_factor: float = 0.3) -> requests.Response:
    """
    Validate URL and download with retry strategy.

    Args:
        url: The URL to download from.
        retries: Number of retry attempts.
        backoff_factor: Backoff factor for retries.

    Returns:
        Response object.

    Raises:
        ValueError: If URL validation fails.
    """
    logger.info(f"DEBUG: validate_and_download_url called with URL: {url[:100]}...")
    parsed = urlparse(url)

    # Validate scheme: only HTTPS allowed
    if parsed.scheme != 'https':
        logger.error(f"DEBUG: URL scheme validation failed: {parsed.scheme} != https")
        raise ValueError("Only HTTPS URLs are allowed")

    # Check for path traversal
    if '..' in parsed.path:
        logger.error(f"DEBUG: Path traversal detected in URL path: {parsed.path}")
        raise ValueError("Path traversal detected in URL")

    # Prevent SSRF: check for localhost and private IPs
    try:
        hostname = parsed.hostname
        if hostname:
            logger.info(f"DEBUG: Resolving hostname: {hostname}")
            ip = socket.gethostbyname(hostname)
            logger.info(f"DEBUG: Hostname resolved to IP: {ip}")
            if (ip.startswith('127.') or
                ip.startswith('10.') or
                (ip.startswith('172.') and 16 <= int(ip.split('.')[1]) <= 31) or
                ip.startswith('192.168.') or
                ip == '0.0.0.0'):
                logger.error(f"DEBUG: IP validation failed - private/internal IP detected: {ip}")
                raise ValueError("Access to internal/private IPs not allowed")
    except socket.gaierror as e:
        # If can't resolve, allow for now (could be a valid external domain)
        logger.warning(f"DEBUG: Could not resolve hostname {parsed.hostname}: {e} - allowing for now")
        pass

    # Use retry strategy
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)

    logger.info(f"DEBUG: Making HTTP request to URL")
    response = session.get(url, stream=True)
    logger.info(f"DEBUG: HTTP response status: {response.status_code}")
    return response


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        # Fallback if psutil not available
        logger.warning("psutil not available, using default memory estimate")
        return 2048  # Assume 2GB available


def calculate_optimal_chunk_size(bbox_area_km2: float, num_bands: int) -> int:
    """Calculate optimal chunk size based on data size and available memory."""
    available_mb = get_available_memory_mb()

    # Estimate memory usage: roughly 4 bytes per pixel per band
    # Conservative estimate: 50% of available memory for processing
    max_processing_mb = available_mb * 0.5

    # Base chunk size on available memory
    chunk_size_mb = min(max_processing_mb / max(num_bands, 1), 256)  # Cap at 256MB

    return max(int(chunk_size_mb), 64)  # Minimum 64MB


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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    item_name: str = ""
) -> bool:
    """
    Download a file from a URL to a local path.

    Args:
        url: The URL to download from.
        output_path: Local path to save the file.
        sign_url: Whether to sign the URL for MPC access.
        overwrite: Whether to overwrite existing files.
        progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes, filename).
        item_name: Name of the item being downloaded for progress reporting.

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

        response = validate_and_download_url(url)
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
                                progress_callback(pbar.n, total_size, item_name or output_path.name)
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
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Path]:
    """
    Download multiple assets from a STAC item.

    Args:
        item_assets: Dictionary of asset key to URL.
        output_dir: Directory to save assets.
        asset_keys: Specific asset keys to download. If None, download all.
        sign_urls: Whether to sign URLs.
        overwrite: Whether to overwrite existing files.
        progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes, filename).

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
        filename = sanitize_filename(Path(urlparse(url).path).name)
        output_path = output_dir / filename

        if download_file(url, output_path, sign_urls, overwrite, progress_callback, item_name=asset_key):
            downloaded_files[asset_key] = output_path
        else:
            logger.error(f"Failed to download asset '{asset_key}'")

    logger.info(f"Downloaded {len(downloaded_files)} assets to {output_dir}")
    return downloaded_files


def download_mtl_files(
    item: pystac.Item,
    output_dir: Union[str, Path],
    mtl_formats: Optional[List[str]] = None,
    overwrite: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Path]:
    """
    Download MTL metadata files from a STAC item without going through clipping logic.

    This function searches for MTL-related assets in the STAC item and downloads them
    directly. It bypasses the clipping/stacking process used for raster data.

    Args:
        item: pystac.Item containing STAC assets
        output_dir: Directory to save MTL files
        mtl_formats: List of MTL file extensions to download (default: ['.txt', '.json', '.xml'])
        overwrite: Whether to overwrite existing files
        progress_callback: Optional callback for progress updates (current, total, filename)

    Returns:
        Dictionary mapping asset keys to downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mtl_formats is None:
        mtl_formats = ['.txt', '.json', '.xml']

    # Define patterns to identify MTL-related assets (case-insensitive)
    mtl_patterns = ['mtl', 'MTL', 'manifest', 'metadata', 'ang', 'qa_radsat', 'qa_aerosol']

    # Preferred media types for MTL files
    preferred_media_types = [
        'text/plain',
        'application/json',
        'application/xml',
        'text/xml'
    ]

    downloaded_files = {}
    total_mtl_files = 0
    found_mtl_files = []

    # First pass: identify MTL-related assets
    for asset_key, asset in item.assets.items():
        asset_key_lower = asset_key.lower()

        # Check if asset key matches MTL patterns
        is_mtl = any(pattern.lower() in asset_key_lower for pattern in mtl_patterns)

        if is_mtl:
            href = getattr(asset, 'href', None)
            media_type = getattr(asset, 'media_type', None)

            if href:
                # Get the filename from the URL
                url_path = urlparse(href).path
                filename = Path(url_path).name

                # Check if filename has a matching extension
                has_valid_extension = any(filename.lower().endswith(ext) for ext in mtl_formats)

                if has_valid_extension:
                    found_mtl_files.append({
                        'key': asset_key,
                        'href': href,
                        'filename': filename,
                        'media_type': media_type
                    })
                    total_mtl_files += 1

    if total_mtl_files == 0:
        logger.info("No MTL files found in STAC item")
        return {}

    logger.info(f"Found {total_mtl_files} MTL files to download")

    # Download each MTL file
    for idx, mtl_info in enumerate(found_mtl_files):
        asset_key = mtl_info['key']
        href = mtl_info['href']
        filename = mtl_info['filename']
        media_type = mtl_info['media_type']

        output_path = output_dir / filename

        # Report progress
        if progress_callback:
            progress_callback(idx, total_mtl_files, filename)

        # Skip if file exists and overwrite is False
        if output_path.exists() and not overwrite:
            logger.info(f"MTL file already exists at {output_path}, skipping download")
            downloaded_files[asset_key] = output_path
            continue

        try:
            # Sign the URL if needed (for MPC access)
            signed_url = href
            if href.startswith("https://"):
                try:
                    signed_url = sign(href)
                except Exception as sign_error:
                    logger.warning(f"Could not sign URL, using unsigned URL: {sign_error}")
                    signed_url = href

            # Validate and download the file
            response = validate_and_download_url(signed_url)
            response.raise_for_status()

            # Write the file
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            downloaded_files[asset_key] = output_path
            logger.info(f"Downloaded MTL file: {output_path} (key: {asset_key})")

        except Exception as e:
            logger.warning(f"Failed to download MTL file '{asset_key}' from {href}: {e}")
            continue

    # Report completion
    if progress_callback:
        progress_callback(total_mtl_files, total_mtl_files, "MTL download completed")

    logger.info(f"Downloaded {len(downloaded_files)} MTL files to {output_dir}")
    return downloaded_files


def download_collection_metadata(
    item: pystac.Item,
    collection: str,
    output_dir: Union[str, Path],
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Path]:
    """
    Download collection-specific metadata files from a STAC item.

    This function downloads metadata files (like MTL, manifest, etc.) for a given collection,
    always overwriting existing files to ensure fresh metadata is always fetched.

    Args:
        item: pystac.Item containing STAC assets
        collection: The STAC collection ID (e.g., 'landsat-c2-l2', 'sentinel-2-l2a')
        output_dir: Directory to save metadata files
        progress_callback: Optional callback for progress updates (current, total, filename)

    Returns:
        Dictionary mapping asset keys to downloaded file paths
    """
    from .utils import get_collection_metadata_assets
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the metadata assets for this collection
    metadata_asset_keys = get_collection_metadata_assets(collection)
    
    if not metadata_asset_keys:
        logger.info(f"No metadata assets configured for collection: {collection}")
        return {}
    
    logger.info(f"Downloading metadata for collection '{collection}': {metadata_asset_keys}")
    
    downloaded_files = {}
    found_metadata = []
    
    # Find matching metadata assets in the item
    for search_key in metadata_asset_keys:
        # Try exact match first, then partial match
        matched_key = None
        
        for item_asset_key in item.assets.keys():
            if search_key.lower() == item_asset_key.lower():
                matched_key = item_asset_key
                break
            elif search_key.lower() in item_asset_key.lower() or item_asset_key.lower() in search_key.lower():
                matched_key = item_asset_key
                logger.info(f"DEBUG: Partial match for metadata: '{search_key}' -> '{item_asset_key}'")
                break
        
        if matched_key:
            href = item.assets[matched_key].href
            # Get filename from URL
            url_path = urlparse(href).path
            filename = Path(url_path).name
            found_metadata.append({
                'key': matched_key,
                'href': href,
                'filename': filename
            })
            logger.info(f"Found metadata asset: '{matched_key}' (search: '{search_key}')")
        else:
            logger.warning(f"No asset found for metadata search key: '{search_key}'")
    
    if not found_metadata:
        logger.info("No metadata files found in STAC item")
        return {}
    
    # Download each metadata file (always overwrite)
    for idx, meta_info in enumerate(found_metadata):
        asset_key = meta_info['key']
        href = meta_info['href']
        filename = meta_info['filename']
        
        output_path = output_dir / filename
        
        # Report progress
        if progress_callback:
            progress_callback(idx, len(found_metadata), filename)
        
        try:
            # Sign the URL if needed (for MPC access)
            signed_url = href
            if href.startswith("https://"):
                try:
                    signed_url = sign(href)
                except Exception as sign_error:
                    logger.warning(f"Could not sign URL, using unsigned: {sign_error}")
                    signed_url = href
            
            # Validate and download
            response = validate_and_download_url(signed_url)
            response.raise_for_status()
            
            # Write the file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            downloaded_files[asset_key] = output_path
            logger.info(f"Downloaded metadata: {output_path} (key: '{asset_key}')")
            
        except Exception as e:
            logger.warning(f"Failed to download metadata '{asset_key}' from {href}: {e}")
            continue
    
    # Report completion
    if progress_callback:
        progress_callback(len(found_metadata), len(found_metadata), "Metadata download completed")
    
    logger.info(f"Downloaded {len(downloaded_files)} metadata files to {output_dir}")
    return downloaded_files


def download_batch(
    urls: List[str],
    output_dir: Union[str, Path],
    filenames: Optional[List[str]] = None,
    sign_urls: bool = True,
    overwrite: bool = False,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
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
        progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes, filename).

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

        item_name = filename or Path(urlparse(url).path).name
        if download_file(url, output_path, sign_urls, overwrite, progress_callback, item_name):
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
    stacked_filename: Optional[str] = None,
    chunk_size_mb: int = 512,  # Memory optimization: process in chunks
    geometry: Optional[Dict] = None,  # GeoJSON geometry for precise clipping
    progress_callback: Optional[Callable[[int, int, str], None]] = None
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
        geometry: Optional GeoJSON geometry for precise clipping (takes precedence over bbox).
        progress_callback: Optional callback for progress updates (bytes_downloaded, total_bytes, filename).

    Returns:
        Dictionary mapping asset keys to output file paths. If stack_bands=True,
        also includes 'stacked' key with the multi-band file path.
    """
    logger.info(f"DEBUG: download_stac_assets_clipped called with {len(asset_keys)} asset keys, bbox: {bbox}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure stac_items is a list
    if not isinstance(stac_items, list):
        stac_items = [stac_items]

    # Filter STAC items to only include those that intersect with the requested bbox
    bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
    intersecting_items = []
    skipped_items = []

    for item in stac_items:
        item_bbox = getattr(item, 'bbox', None)
        if item_bbox is None:
            logger.warning(f"STAC item {getattr(item, 'id', 'unknown')} has no bbox, skipping")
            skipped_items.append(getattr(item, 'id', 'unknown'))
            continue

        item_geom = box(item_bbox[0], item_bbox[1], item_bbox[2], item_bbox[3])
        if bbox_geom.intersects(item_geom):
            intersecting_items.append(item)
        else:
            item_id = getattr(item, 'id', 'unknown')
            logger.warning(f"STAC item {item_id} does not intersect with requested bbox {bbox}, skipping")
            logger.warning(f"Item bbox: {item_bbox}")
            skipped_items.append(item_id)

    # Ensure at least one item intersects
    if not intersecting_items:
        error_msg = f"No STAC items intersect with the requested bbox {bbox}. "
        if skipped_items:
            error_msg += f"Skipped items: {skipped_items}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Filtered STAC items: {len(intersecting_items)} intersecting items out of {len(stac_items)} total")
    stac_items = intersecting_items

    # DEBUG: Comprehensive logging of STAC items and assets before processing
    logger.info(f"DEBUG: Number of STAC items received: {len(stac_items)}")
    for i, item in enumerate(stac_items):
        item_id = getattr(item, 'id', 'N/A')
        item_bbox = getattr(item, 'bbox', 'N/A')
        item_datetime = getattr(item, 'datetime', 'N/A')
        logger.info(f"DEBUG: Item {i} - ID: {item_id}, bbox: {item_bbox}, datetime: {item_datetime}")
        assets = getattr(item, 'assets', {})
        logger.info(f"DEBUG: Item {i} has {len(assets)} assets:")
        for asset_key, asset in assets.items():
            href = getattr(asset, 'href', 'N/A')
            media_type = getattr(asset, 'media_type', 'N/A')
            roles = getattr(asset, 'roles', [])
            logger.info(f"DEBUG:   Asset '{asset_key}': href={href[:100]}..., media_type={media_type}, roles={roles}")

    logger.info(f"DEBUG: Requested asset_keys (bands): {asset_keys}")

    # Check for items with no assets
    items_with_no_assets = [i for i, item in enumerate(stac_items) if not getattr(item, 'assets', {})]
    if items_with_no_assets:
        logger.warning(f"DEBUG: Items with no assets: {items_with_no_assets}")

    # Check if requested bands exist in items
    missing_bands = []
    for asset_key in asset_keys:
        found_in_all = True
        for item in stac_items:
            if asset_key not in getattr(item, 'assets', {}):
                found_in_all = False
                break
        if not found_in_all:
            missing_bands.append(asset_key)
    if missing_bands:
        logger.warning(f"DEBUG: Requested bands not found in some/all items: {missing_bands}")

    # Validate STAC items
    if not stac_items:
        logger.error("DEBUG: No STAC items provided to download_stac_assets_clipped")
        return {}

    logger.info(f"DEBUG: Processing {len(stac_items)} STAC items")
    for i, item in enumerate(stac_items):
        if not hasattr(item, 'assets') or not item.assets:
            logger.error(f"DEBUG: STAC item {i} has no assets or invalid assets attribute")
            logger.error(f"DEBUG: Item type: {type(item)}, Item attributes: {dir(item)}")
            continue
        logger.info(f"DEBUG: Item {i} has {len(item.assets)} assets: {list(item.assets.keys())}")

    # Separate asset_keys into raster and metadata assets
    raster_asset_keys = []
    metadata_asset_keys = []
    logger.info(f"DEBUG: Processing {len(asset_keys)} asset keys: {asset_keys}")
    logger.info(f"DEBUG: Available assets in first item: {list(stac_items[0].assets.keys())}")
    
    # Log all assets and their media types for debugging
    logger.info("DEBUG: Detailed asset analysis:")
    for key, asset in stac_items[0].assets.items():
        media_type = getattr(asset, 'media_type', None)
        href = getattr(asset, 'href', 'N/A')
        roles = getattr(asset, 'roles', [])
        logger.info(f"DEBUG: Asset '{key}': media_type='{media_type}', roles={roles}, href={href[:100]}...")

    # Log metadata asset keys being searched for
    logger.info(f"DEBUG: Searching for metadata assets: {asset_keys}")
    
    # Find matching metadata assets by checking if any asset key contains the search pattern
    for search_key in asset_keys:
        # Check if any asset key contains the search pattern (case-insensitive)
        matching_assets = [k for k in stac_items[0].assets.keys() if search_key.lower() in k.lower()]
        if matching_assets:
            logger.info(f"DEBUG: Found match for '{search_key}': {matching_assets}")
        else:
            logger.warning(f"DEBUG: No assets found matching '{search_key}'")

    for asset_key in asset_keys:
        # Check if asset exists in the first item (exact match or partial match for metadata)
        asset_found = False
        matched_key = None
        
        # First try exact match
        if asset_key in stac_items[0].assets:
            asset_found = True
            matched_key = asset_key
        else:
            # Try partial match for metadata files (e.g., 'mtl.txt' should match 'MTL')
            for item_asset_key in stac_items[0].assets.keys():
                if asset_key.lower() in item_asset_key.lower() or item_asset_key.lower() in asset_key.lower():
                    asset_found = True
                    matched_key = item_asset_key
                    logger.info(f"DEBUG: Partial match found: '{asset_key}' -> '{item_asset_key}'")
                    break
        
        if asset_found and matched_key:
            asset = stac_items[0].assets[matched_key]
            media_type = getattr(asset, 'media_type', None)
            href = getattr(asset, 'href', 'N/A')
            roles = getattr(asset, 'roles', [])
            logger.info(f"DEBUG: Processing requested asset '{asset_key}' (matched: '{matched_key}'): media_type='{media_type}', roles={roles}, href={href[:100]}...")
            # Check if asset is a raster asset
            is_raster = False
            if media_type and media_type.startswith('image/'):
                is_raster = True
                logger.info(f"DEBUG: '{asset_key}' is raster (media_type starts with 'image/')")
            elif media_type and ('tiff' in media_type.lower() or 'geotiff' in media_type.lower()):
                is_raster = True
                logger.info(f"DEBUG: '{asset_key}' is raster (media_type contains 'tiff' or 'geotiff')")
            elif roles and 'data' in roles and media_type:
                # Additional check for data role assets that might be raster
                is_raster = True
                logger.info(f"DEBUG: '{asset_key}' is raster (has 'data' role and media_type='{media_type}')")
            elif not media_type and roles and 'data' in roles:
                # Fallback for assets with no media_type but data role
                is_raster = True
                logger.info(f"DEBUG: '{asset_key}' assumed raster (no media_type but has 'data' role)")

            if is_raster:
                raster_asset_keys.append(matched_key)  # Use actual asset key
                logger.info(f"DEBUG: Added '{matched_key}' to raster assets")
            else:
                metadata_asset_keys.append(matched_key)  # Use actual asset key
                logger.info(f"DEBUG: Added '{matched_key}' to metadata assets")
        else:
            logger.warning(f"DEBUG: Requested asset key '{asset_key}' not found in item assets")
            logger.warning(f"DEBUG: Available asset keys: {list(stac_items[0].assets.keys())}")
    
    logger.info(f"DEBUG: Final raster assets: {raster_asset_keys}")
    logger.info(f"DEBUG: Final metadata assets: {metadata_asset_keys}")

    # Initialize downloaded_files dictionary (will hold both metadata and raster files)
    downloaded_files = {}

    # Download metadata assets without processing (always overwrite)
    if metadata_asset_keys:
        if progress_callback:
            progress_callback(0, 100, "Downloading metadata files")
        signed_urls = {}
        for key in metadata_asset_keys:
            if key in stac_items[0].assets:
                signed_urls[key] = stac_items[0].assets[key].href
        logger.info(f"DEBUG: Metadata download - keys: {metadata_asset_keys}, overwrite={overwrite}")
        metadata_files = download_stac_assets(signed_urls, output_dir, metadata_asset_keys, sign_urls=False, overwrite=overwrite, progress_callback=progress_callback)  # Already signed
        logger.info(f"DEBUG: Metadata download result: {metadata_files}")
        downloaded_files.update(metadata_files)

    asset_keys = raster_asset_keys
    if not asset_keys:
        logger.error("DEBUG: No valid raster assets found to process - this will cause empty reader_table error")
        logger.error(f"DEBUG: Requested asset keys: {asset_keys}")
        logger.error(f"DEBUG: Available assets: {list(stac_items[0].assets.keys())}")
        
        # Log asset details for debugging
        for key, asset in stac_items[0].assets.items():
            media_type = getattr(asset, 'media_type', 'N/A')
            logger.error(f"DEBUG: Asset '{key}' media_type: '{media_type}' (starts with 'image/'? {media_type.startswith('image/') if media_type else False})")
        
        logger.info("No valid raster assets found to process, but metadata downloaded")
        return downloaded_files

    # Extract acquisition date
    date_obj = stac_items[0].datetime
    date_str = date_obj.strftime('%Y%m%d') if date_obj else 'nodate'

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
    logger.info(f"DEBUG: Calculated UTM EPSG: {utm_epsg} from bbox center")

    # Calculate bounds in UTM for accurate clipping
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    min_x, min_y = transformer.transform(bbox[0], bbox[1])
    max_x, max_y = transformer.transform(bbox[2], bbox[3])
    bounds = (min_x, min_y, max_x, max_y)
    logger.info(f"DEBUG: Calculated UTM bounds: {bounds} from WGS84 bbox {bbox}")
    logger.info(f"DEBUG: Transformer used: EPSG:4326 -> EPSG:{utm_epsg}")

    # Calculate optimal chunk size based on data size and available memory
    if chunk_size_mb == 512:  # Default value, calculate optimal
        # Estimate area in km² for chunk size calculation
        from math import radians, sin, cos, sqrt
        lat1, lon1, lat2, lon2 = map(radians, bbox)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * 6371 * 6371 * a  # Earth's radius squared
        bbox_area_km2 = abs(c) if c > 0 else 1000  # Fallback to 1000 km²

        chunk_size_mb = calculate_optimal_chunk_size(bbox_area_km2, len(asset_keys))
        logger.info(f"Calculated optimal chunk size: {chunk_size_mb} MB for {bbox_area_km2:.0f} km² area")

    # Validate bounds are reasonable
    min_x, min_y, max_x, max_y = bounds
    if max_x <= min_x or max_y <= min_y:
        logger.error(f"DEBUG: Invalid UTM bounds: {bounds} - max_x <= min_x or max_y <= min_y")
        raise ValueError(f"Invalid UTM bounds calculated: {bounds}")

    # Check if bounds are too small (less than 1 pixel at 10m resolution)
    width_m = max_x - min_x
    height_m = max_y - min_y
    if width_m < 10 or height_m < 10:
        logger.warning(f"DEBUG: UTM bounds very small: {width_m:.1f}m x {height_m:.1f}m - may result in empty data")
        logger.warning(f"DEBUG: This could cause 'Empty reader_table' if no pixels fall within bounds")

    # Stack the items with memory optimization
    if progress_callback:
        progress_callback(0, 100, "Stacking raster data")
    
    # DEBUG: Log detailed information about UTM bounds and asset bounds
    logger.info(f"DEBUG: === UTM BOUNDS ANALYSIS ===")
    logger.info(f"DEBUG: Calculated UTM bounds: {bounds}")
    logger.info(f"DEBUG: UTM EPSG: {utm_epsg}")
    logger.info(f"DEBUG: Original WGS84 bbox: {bbox}")
    
    # DEBUG: Log bounds of each individual asset
    logger.info(f"DEBUG: === INDIVIDUAL ASSET BOUNDS ===")
    for i, item in enumerate(stac_items):
        item_id = getattr(item, 'id', f'item_{i}')
        item_bbox = getattr(item, 'bbox', None)
        logger.info(f"DEBUG: Item {i} ({item_id}) WGS84 bbox: {item_bbox}")
    
        # Get bounds for each asset in this item
        for asset_key in asset_keys:
            if asset_key in item.assets:
                asset = item.assets[asset_key]
                # Try to get asset bounds from various sources
                asset_bbox = getattr(asset, 'bbox', None)
                asset_geometry = getattr(asset, 'geometry', None)
                logger.info(f"DEBUG:   Asset '{asset_key}' WGS84 bbox: {asset_bbox}")
                logger.info(f"DEBUG:   Asset '{asset_key}' geometry: {asset_geometry}")
    
    # DEBUG: Check if UTM bounds intersect with asset bounds
    logger.info(f"DEBUG: === BOUNDS INTERSECTION VALIDATION ===")
    
    # Convert UTM bounds back to WGS84 for comparison
    reverse_transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    utm_min_x, utm_min_y, utm_max_x, utm_max_y = bounds
    
    # Convert UTM corners to WGS84
    wgs84_min_lon, wgs84_min_lat = reverse_transformer.transform(utm_min_x, utm_min_y)
    wgs84_max_lon, wgs84_max_lat = reverse_transformer.transform(utm_max_x, utm_max_y)
    
    # Create shapely geometry for UTM bounds in WGS84
    utm_bounds_wgs84 = box(wgs84_min_lon, wgs84_min_lat, wgs84_max_lon, wgs84_max_lat)
    logger.info(f"DEBUG: UTM bounds converted back to WGS84: {utm_bounds_wgs84.bounds}")
    
    # Check intersection with each item's bbox
    all_items_intersect = True
    for i, item in enumerate(stac_items):
        item_bbox = getattr(item, 'bbox', None)
        if item_bbox:
            item_geom = box(item_bbox[0], item_bbox[1], item_bbox[2], item_bbox[3])
            intersects = utm_bounds_wgs84.intersects(item_geom)
            logger.info(f"DEBUG: Item {i} bbox intersects UTM bounds (WGS84): {intersects}")
            if not intersects:
                all_items_intersect = False
                logger.warning(f"DEBUG: WARNING: Item {i} does not intersect with UTM bounds!")
                logger.warning(f"DEBUG: Item bbox: {item_bbox}")
                logger.warning(f"DEBUG: UTM bounds (WGS84): {utm_bounds_wgs84.bounds}")
    
    # DEBUG: Validate UTM bounds are reasonable
    logger.info(f"DEBUG: === UTM BOUNDS VALIDATION ===")
    min_x, min_y, max_x, max_y = bounds
    width_m = max_x - min_x
    height_m = max_y - min_y
    area_m2 = width_m * height_m
    logger.info(f"DEBUG: UTM bounds dimensions: {width_m:.1f}m x {height_m:.1f}m = {area_m2:.1f}m²")
    
    # Check if bounds are too small
    if width_m < resolution or height_m < resolution:
        logger.error(f"DEBUG: ERROR: UTM bounds too small for resolution!")
        logger.error(f"DEBUG: Width {width_m:.1f}m < resolution {resolution}m")
        logger.error(f"DEBUG: Height {height_m:.1f}m < resolution {resolution}m")
        logger.error(f"DEBUG: This will result in empty reader_table!")
    
    # Check if bounds are valid (not inverted)
    if max_x <= min_x or max_y <= min_y:
        logger.error(f"DEBUG: ERROR: Invalid UTM bounds - max <= min!")
        logger.error(f"DEBUG: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        logger.error(f"DEBUG: This will cause empty reader_table!")
    
    # Check if bounds are within reasonable UTM range
    if abs(min_x) > 1e6 or abs(max_x) > 1e6 or abs(min_y) > 1e7 or abs(max_y) > 1e7:
        logger.warning(f"DEBUG: WARNING: UTM bounds seem unusually large!")
        logger.warning(f"DEBUG: This might indicate coordinate system issues")
    
    logger.info(f"DEBUG: About to call stackstac.stack with:")
    logger.info(f"  - stac_items: {len(stac_items)} items")
    logger.info(f"  - assets: {asset_keys} (filtered raster assets)")
    logger.info(f"  - resolution: {resolution}")
    logger.info(f"  - epsg: {utm_epsg}")
    logger.info(f"  - bounds: {bounds}")
    logger.info(f"  - chunksize: {chunk_size_mb * 1024 * 1024} bytes")

    # Additional validation before stackstac call
    if not asset_keys:
        logger.error("DEBUG: No raster assets available for stackstac.stack() - this will cause empty reader_table")
        logger.error("DEBUG: Check asset filtering logic above for why no assets were selected")
        raise ValueError("No valid raster assets found for stacking")

    # Validate that all asset_keys exist in all items
    for i, item in enumerate(stac_items):
        missing_assets = [key for key in asset_keys if key not in item.assets]
        if missing_assets:
            logger.warning(f"DEBUG: Item {i} missing assets: {missing_assets}")
            logger.warning(f"DEBUG: Item {i} available assets: {list(item.assets.keys())}")
    
    try:
        stack = stackstac.stack(
            stac_items,
            assets=asset_keys,
            resolution=resolution,
            epsg=utm_epsg,
            bounds=bounds,
            chunksize=chunk_size_mb * 1024 * 1024  # Convert MB to bytes for chunking
        )
        logger.info(f"DEBUG: After stackstac.stack - CRS: {stack.rio.crs}, Transform: {stack.rio.transform()}")
        logger.info(f"DEBUG: Stack shape: {stack.shape}, dims: {stack.dims}")

        # DEBUG: Detailed analysis of stackstac result
        logger.info(f"DEBUG: === STACKSTAC RESULT ANALYSIS ===")
        logger.info(f"DEBUG: Stack type: {type(stack)}")
        logger.info(f"DEBUG: Stack attributes: {dir(stack)}")

        # Check if stack has any data
        if hasattr(stack, 'size') and stack.size == 0:
            logger.error(f"DEBUG: Stack is empty - size: {stack.size}")
            raise ValueError(f"Empty stack created - no data found for bounds {bounds}")

        # Check for reader_table issue specifically
        if hasattr(stack, '_reader_table'):
            logger.info(f"DEBUG: Stack has _reader_table attribute")
            logger.info(f"DEBUG: reader_table type: {type(stack._reader_table)}")
            logger.info(f"DEBUG: reader_table attributes: {dir(stack._reader_table)}")

            if hasattr(stack._reader_table, 'shape'):
                logger.info(f"DEBUG: reader_table.shape: {stack._reader_table.shape}")
                if stack._reader_table.shape == (0, 0):
                    logger.error(f"DEBUG: Empty reader_table detected with shape (0, 0)")
                    logger.error(f"DEBUG: This indicates no valid raster data was found for the given bounds and assets")
                    logger.error(f"DEBUG: Possible causes:")
                    logger.error(f"  - Bounds {bounds} do not intersect with actual data extent")
                    logger.error(f"  - Assets {asset_keys} are not valid raster files")
                    logger.error(f"  - Assets are not accessible or corrupted")
                    logger.error(f"  - UTM bounds calculation may be incorrect")
                    logger.error(f"DEBUG: Item bbox: {stac_items[0].bbox}")
                    logger.error(f"DEBUG: Requested bbox: {bbox}")
                    logger.error(f"DEBUG: Calculated UTM bounds: {bounds}")

                    # Additional debugging: log asset URLs
                    logger.error(f"DEBUG: === ASSET URLs FOR DEBUGGING ===")
                    for i, item in enumerate(stac_items):
                        for asset_key in asset_keys:
                            if asset_key in item.assets:
                                asset = item.assets[asset_key]
                                href = getattr(asset, 'href', 'N/A')
                                logger.error(f"DEBUG: Item {i} asset '{asset_key}': {href}")

                    raise ValueError("Empty reader_table: reader_table.shape=(0, 0) - no valid raster data found")

            # Check if reader_table has any rows
            if hasattr(stack._reader_table, '__len__'):
                reader_table_len = len(stack._reader_table)
                logger.info(f"DEBUG: reader_table length: {reader_table_len}")
                if reader_table_len == 0:
                    logger.error(f"DEBUG: Empty reader_table detected - length: 0")
                    logger.error(f"DEBUG: This indicates no valid raster data was found")
                    raise ValueError("Empty reader_table: length=0 - no valid raster data found")

        else:
            logger.warning(f"DEBUG: Stack does not have _reader_table attribute")
            logger.warning(f"DEBUG: This might be a different stackstac version or configuration")

        # Additional validation: check if stack has any non-null values
        try:
            if hasattr(stack, 'count'):
                non_null_count = stack.count().compute()
                logger.info(f"DEBUG: Non-null pixel count per band: {non_null_count}")
                if hasattr(non_null_count, 'sum') and non_null_count.sum() == 0:
                    logger.error(f"DEBUG: All pixels are null/empty in the stack!")
                    logger.error(f"DEBUG: This indicates bounds do not intersect with actual data")
                    raise ValueError("All pixels are null - bounds do not intersect with data")
        except Exception as e:
            logger.warning(f"DEBUG: Could not check non-null pixel count: {e}")

        # Final validation: ensure we have valid data before proceeding
        if hasattr(stack, 'size') and stack.size == 0:
            logger.error(f"DEBUG: Final validation failed - stack is empty")
            raise ValueError("Final validation: stack is empty after all processing")

    except Exception as e:
        logger.error(f"DEBUG: stackstac.stack failed: {e}")
        logger.error(f"DEBUG: This might be the source of the 'Empty reader_table' error")

        # Additional debugging for stackstac failure
        logger.error(f"DEBUG: === STACKSTAC FAILURE ANALYSIS ===")
        logger.error(f"DEBUG: Failed with bounds: {bounds}")
        logger.error(f"DEBUG: Failed with UTM EPSG: {utm_epsg}")
        logger.error(f"DEBUG: Failed with assets: {asset_keys}")

        # Log asset bounds in UTM for comparison
        logger.error(f"DEBUG: === ASSET BOUNDS IN UTM FOR COMPARISON ===")
        for i, item in enumerate(stac_items):
            item_id = getattr(item, 'id', f'item_{i}')
            item_bbox = getattr(item, 'bbox', None)
            if item_bbox:
                # Convert item bbox to UTM
                item_min_x, item_min_y = transformer.transform(item_bbox[0], item_bbox[1])
                item_max_x, item_max_y = transformer.transform(item_bbox[2], item_bbox[3])
                item_utm_bounds = (item_min_x, item_min_y, item_max_x, item_max_y)
                logger.error(f"DEBUG: Item {i} ({item_id}) UTM bounds: {item_utm_bounds}")

                # Check intersection with requested bounds
                item_geom = box(item_utm_bounds[0], item_utm_bounds[1], item_utm_bounds[2], item_utm_bounds[3])
                bounds_geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
                intersects = item_geom.intersects(bounds_geom)
                logger.error(f"DEBUG: Item {i} UTM bounds intersect requested bounds: {intersects}")

        raise

    # Apply geometry-based clipping if geometry is provided
    if geometry:
        from shapely.geometry import shape
        try:
            geom = shape(geometry)
            # Convert geometry to the same CRS as the stack (UTM)
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
            geom_utm = transformer.transform_geom(geom)

            # Create mask from geometry
            import numpy as np
            from rasterio.features import geometry_mask

            # Get stack dimensions and transform
            height, width = stack.shape[-2], stack.shape[-1]
            transform = stack.rio.transform()

            # Create geometry mask
            mask = geometry_mask([geom_utm], (height, width), transform, invert=True)

            # Apply mask to stack
            stack = stack.where(mask, np.nan)
            logger.info("Applied geometry-based clipping to stack")
            logger.info(f"DEBUG: After geometry clipping - CRS: {stack.rio.crs}, Transform: {stack.rio.transform()}")

        except Exception as e:
            logger.warning(f"Failed to apply geometry clipping, proceeding with bbox clipping: {e}")

    # Compute the stack with memory-efficient processing
    if progress_callback:
        progress_callback(0, 100, "Computing raster data")
    if show_progress:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            # Use compute with memory limit to prevent out-of-memory errors
            with dask.config.set(scheduler='threads', num_workers=2):  # Limit threads for memory
                stack = stack.compute()
    else:
        # Use synchronous scheduler for better memory control
        with dask.config.set(scheduler='synchronous'):
            stack = stack.compute()
    logger.info(f"DEBUG: After compute - CRS: {stack.rio.crs}, Transform: {stack.rio.transform()}")

    # Save files based on stack_bands option
    # Note: downloaded_files already contains metadata files from above, so we preserve them

    if stack_bands:
        # Only create stacked multi-band TIFF
        if stacked_filename is None:
            # Use item ID if available, otherwise generic name
            if hasattr(stac_items[0], 'id'):
                stacked_filename = f"{stac_items[0].id}_{date_str}_stacked.tif"
            else:
                stacked_filename = f"stacked_bands_{date_str}.tif"

        # Sanitize the filename
        stacked_filename = sanitize_filename(stacked_filename)
        
        # Ensure output directory exists and is valid
        try:
            output_dir = Path(output_dir)
            logger.debug(f"DEBUG: Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate the directory path
            if not output_dir.exists() or not output_dir.is_dir():
                raise ValueError(f"Cannot create or access directory: {output_dir}")
                
            logger.debug(f"DEBUG: Output directory validated: {output_dir}")
            
        except Exception as e:
            logger.error(f"DEBUG: Failed to create/validate output directory {output_dir}: {e}")
            raise

        stacked_path = output_dir / stacked_filename
        logger.debug(f"DEBUG: Final stacked file path: {stacked_path}")
        
        if stacked_path.exists() and not overwrite:
            logger.info(f"Stacked file {stacked_path} already exists, skipping")
        else:
            if progress_callback:
                progress_callback(0, 100, f"Saving stacked file: {stacked_filename}")
            # Prepare the stack for rioxarray - ensure it's (band, y, x)
            # Remove time dimension if it exists (should be 1)
            if 'time' in stack.dims:
                stack_for_save = stack.isel(time=0)
            else:
                stack_for_save = stack

            # Ensure band dimension is first
            if stack_for_save.dims[0] != 'band':
                stack_for_save = stack_for_save.transpose('band', ...)

            logger.info(f"DEBUG: Before saving stacked - CRS: {stack_for_save.rio.crs}, Transform: {stack_for_save.rio.transform()}")
            
            try:
                stack_for_save.rio.to_raster(str(stacked_path))
                logger.info(f"Saved stacked multi-band TIFF to {stacked_path}")
            except Exception as e:
                logger.error(f"DEBUG: Failed to save stacked file to {stacked_path}: {e}")
                logger.error(f"DEBUG: Path exists: {stacked_path.exists()}, Is dir: {stacked_path.is_dir()}")
                logger.error(f"DEBUG: Parent dir exists: {stacked_path.parent.exists()}, Is writable: {os.access(stacked_path.parent, os.W_OK)}")
                raise

        downloaded_files['stacked'] = stacked_path
    else:
        # Save each band as separate GeoTIFF
        total_bands = len(stack.band.values)
        
        # Ensure output directory is valid before processing bands
        try:
            output_dir = Path(output_dir)
            if not output_dir.exists() or not output_dir.is_dir():
                output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"DEBUG: Failed to create output directory for bands: {e}")
            raise
        
        for i, band in enumerate(stack.band.values):
            band_data = stack.sel(band=band)
            
            # Create sanitized filename
            band_filename = f"{band}_{date_str}.tif"
            band_filename = sanitize_filename(band_filename)
            output_path = output_dir / band_filename
            
            logger.debug(f"DEBUG: Processing band {band}, output path: {output_path}")
            
            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path} already exists, skipping")
                downloaded_files[str(band)] = output_path
                continue
            if progress_callback:
                progress_callback(int((i / total_bands) * 100), 100, f"Saving band {band}")
            logger.info(f"DEBUG: Before saving band {band} - CRS: {band_data.rio.crs}, Transform: {band_data.rio.transform()}")
            
            try:
                band_data.rio.to_raster(str(output_path))
                logger.info(f"Saved clipped asset '{band}' to {output_path}")
                downloaded_files[str(band)] = output_path
            except Exception as e:
                logger.error(f"DEBUG: Failed to save band {band} to {output_path}: {e}")
                logger.error(f"DEBUG: Path validation - exists: {output_path.exists()}, parent writable: {os.access(output_path.parent, os.W_OK)}")
                raise


    if progress_callback:
        progress_callback(100, 100, "Processing completed")
    logger.info(f"Downloaded and clipped {len(downloaded_files)} assets to {output_dir}")
    return downloaded_files
