"""
STAC module

This module provides utilities for working with STAC (SpatioTemporal Asset Catalog) data.

"""

import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pystac
from pystac_client import Client
from planetary_computer import sign
import stackstac
import geopandas as gpd
from shapely.geometry import shape, Polygon

logger = logging.getLogger(__name__)


class MPCSTACClient:
    """
    A client for interacting with Microsoft Planetary Computer STAC API.

    This class provides methods to connect to the MPC STAC catalog,
    search for items, and handle signed URLs for data access.
    """

    MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, api_url: Optional[str] = None):
        """
        Initialize the MPC STAC client.

        Args:
            api_url: Optional custom STAC API URL. Defaults to MPC STAC URL.
        """
        self.api_url = api_url or self.MPC_STAC_URL
        self.client = Client.open(self.api_url)
        logger.info(f"Connected to STAC API at {self.api_url}")

    def search_items(
        self,
        collection: str,
        bbox: Optional[List[float]] = None,
        datetime_range: Optional[str] = None,
        limit: int = 100,
        **kwargs
    ) -> List[pystac.Item]:
        """
        Search for STAC items in a collection.

        Args:
            collection: Collection ID to search in.
            bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat].
            datetime_range: Date range in ISO 8601 format (e.g., "2020-01-01/2020-12-31").
            limit: Maximum number of items to return.
            **kwargs: Additional search parameters.

        Returns:
            List of matching STAC items.
        """
        search_params = {
            "collections": [collection],
            "limit": limit,
            **kwargs
        }

        if bbox:
            search_params["bbox"] = bbox
        if datetime_range:
            search_params["datetime"] = datetime_range

        search = self.client.search(**search_params)
        items = list(search.items())
        logger.info(f"Found {len(items)} items in collection '{collection}'")
        return items

    def get_item(self, item_id: str, collection: str) -> Optional[pystac.Item]:
        """
        Retrieve a specific STAC item by ID.

        Args:
            item_id: The item ID.
            collection: The collection containing the item.

        Returns:
            The STAC item if found, None otherwise.
        """
        try:
            item = self.client.get_item(item_id, collection=collection)
            logger.info(f"Retrieved item '{item_id}' from collection '{collection}'")
            return item
        except Exception as e:
            logger.error(f"Failed to retrieve item '{item_id}': {e}")
            return None

    def get_collection_bands(self, collection: str) -> List[str]:
        """
        Get available band names for a collection.

        Args:
            collection: Collection ID to get bands for.

        Returns:
            List of band names available in the collection.
        """
        try:
            # Get the collection
            collection_obj = self.client.get_collection(collection)
            
            # Get a sample item to extract band information
            search = self.client.search(collections=[collection], limit=1)
            items = list(search.items())
            
            if not items:
                logger.warning(f"No items found in collection '{collection}'")
                return []
            
            # Get band names from the first item's assets
            sample_item = items[0]
            band_names = []
            
            for asset_key, asset in sample_item.assets.items():
                # Filter out non-band assets (like metadata, thumbnails, etc.)
                if hasattr(asset, 'eo_bands') and asset.eo_bands:
                    # This is an EO band asset
                    for band in asset.eo_bands:
                        if band.name and band.name not in band_names:
                            band_names.append(band.name)
                elif asset_key not in ['metadata', 'thumbnail', 'overview']:
                    # Add asset key as band name for non-EO assets
                    band_names.append(asset_key)
            
            logger.info(f"Found {len(band_names)} bands in collection '{collection}': {band_names}")
            return band_names
            
        except Exception as e:
            logger.error(f"Failed to get bands for collection '{collection}': {e}")
            return []

    def get_signed_urls(self, item: pystac.Item) -> Dict[str, str]:
        """
        Get signed URLs for all assets in a STAC item.

        Args:
            item: The STAC item containing assets.

        Returns:
            Dictionary mapping asset keys to signed URLs.
        """
        signed_urls = {}
        for asset_key, asset in item.assets.items():
            if asset.href.startswith("https://"):
                signed_url = sign(asset.href)
                signed_urls[asset_key] = signed_url
            else:
                signed_urls[asset_key] = asset.href
        logger.debug(f"Generated signed URLs for {len(signed_urls)} assets")
        return signed_urls

    def items_to_geodataframe(self, items: List[pystac.Item]) -> gpd.GeoDataFrame:
        """
        Convert STAC items to a GeoDataFrame.

        Args:
            items: List of STAC items.

        Returns:
            GeoDataFrame with item properties and geometries.
        """
        features = []
        for item in items:
            feature = {
                "id": item.id,
                "bbox": item.bbox,
                "datetime": item.datetime,
                "geometry": shape(item.geometry),
                **item.properties
            }
            features.append(feature)

        gdf = gpd.GeoDataFrame(features)
        logger.info(f"Converted {len(items)} items to GeoDataFrame")
        return gdf


def search_mpc_collection(
    collection: str,
    bbox: Optional[List[float]] = None,
    datetime_range: Optional[str] = None,
    limit: int = 100,
    path: Optional[int] = None,
    row: Optional[int] = None,
    mgrs_tile: Optional[str] = None,
    require_full_bbox_coverage: bool = False
) -> List[pystac.Item]:
    """
    Convenience function to search items in an MPC collection.

    Args:
        collection: Collection ID.
        bbox: Bounding box.
        datetime_range: Date range.
        limit: Max items.
        path: WRS path number (for Landsat collections).
        row: WRS row number (for Landsat collections).
        mgrs_tile: MGRS tile ID (for Sentinel-2 collections).
        require_full_bbox_coverage: If True, only return items whose footprint fully covers the bbox.

    Returns:
        List of STAC items.
    """
    client = MPCSTACClient()
    query = {}
    if path is not None:
        query["landsat:wrs_path"] = {"eq": path}
    if row is not None:
        query["landsat:wrs_row"] = {"eq": row}
    if mgrs_tile is not None:
        query["s2:mgrs_tile"] = {"eq": mgrs_tile}
    kwargs = {}
    if query:
        kwargs["query"] = query
    items = client.search_items(collection, bbox, datetime_range, limit, **kwargs)

    if require_full_bbox_coverage and bbox:
        # Create bbox polygon
        bbox_poly = Polygon([
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3])
        ])
        # Filter items where footprint fully contains bbox
        filtered_items = []
        for item in items:
            item_geom = shape(item.geometry)
            if item_geom.contains(bbox_poly):
                filtered_items.append(item)
        logger.info(f"Filtered to {len(filtered_items)} items with full bbox coverage")
        return filtered_items

    return items


def get_mpc_item(item_id: str, collection: str) -> Optional[pystac.Item]:
    """
    Convenience function to get a specific MPC item.

    Args:
        item_id: Item ID.
        collection: Collection ID.

    Returns:
        STAC item if found.
    """
    client = MPCSTACClient()
    return client.get_item(item_id, collection)


def get_mpc_collection_bands(collection: str) -> List[str]:
    """
    Convenience function to get available bands for an MPC collection.

    Args:
        collection: Collection ID.

    Returns:
        List of band names available in the collection.
    """
    client = MPCSTACClient()
    return client.get_collection_bands(collection)