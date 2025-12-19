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
from shapely.geometry import shape

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
    limit: int = 100
) -> List[pystac.Item]:
    """
    Convenience function to search items in an MPC collection.

    Args:
        collection: Collection ID.
        bbox: Bounding box.
        datetime_range: Date range.
        limit: Max items.

    Returns:
        List of STAC items.
    """
    client = MPCSTACClient()
    return client.search_items(collection, bbox, datetime_range, limit)


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