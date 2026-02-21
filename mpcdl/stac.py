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
            # Search for the item in the specific collection
            search = self.client.search(collections=[collection], ids=[item_id], limit=1)
            items = list(search.items())
            if items:
                item = items[0]
                logger.info(f"Retrieved item '{item_id}' from collection '{collection}'")
                return item
            else:
                logger.warning(f"Item '{item_id}' not found in collection '{collection}'")
                return None
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

    def get_item_assets(self, item: pystac.Item) -> List[str]:
        """
        Get available asset keys from a specific STAC item.

        Args:
            item: The STAC item to get assets from.

        Returns:
            List of asset keys available in the item.
        """
        if not item or not hasattr(item, 'assets'):
            logger.warning("Item has no assets")
            return []
        
        assets = list(item.assets.keys())
        logger.info(f"Retrieved {len(assets)} assets from item '{item.id}': {', '.join(assets)}")
        return assets

    def get_item_assets_detailed(self, item: pystac.Item) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all assets in a STAC item.

        This function extracts comprehensive information about each asset in a STAC item,
        including URLs, media types, roles, descriptions, and additional metadata like
        file size, spatial resolution, and bounding box when available.

        Args:
            item (pystac.Item): The STAC item containing assets.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping asset keys to detailed asset information.
                Each asset info dict contains:
                - 'url': Asset URL
                - 'type': Media type
                - 'roles': List of roles (e.g., ['data', 'thumbnail'])
                - 'description': Asset description or title
                - 'file_size': File size in bytes (if available)
                - 'gsd': Ground sample distance/resolution (if available)
                - 'bbox': Asset bounding box (if available)
                - 'shape': Image dimensions (if available)
        """
        assets_info = {}

        for asset_key, asset in item.assets.items():
            asset_info = {
                "url": asset.href,
                "type": asset.media_type or "unknown",
                "roles": list(asset.roles) if hasattr(asset, 'roles') and asset.roles else [],
                "description": "",
                "file_size": None,
                "gsd": None,
                "bbox": None,
                "shape": None
            }

            # Extract description and additional metadata
            if hasattr(asset, 'properties') and asset.properties:
                asset_info["description"] = asset.properties.get('title') or asset.properties.get('description') or ""
                asset_info["file_size"] = asset.properties.get('file:size')
                asset_info["gsd"] = asset.properties.get('gsd')
                asset_info["bbox"] = asset.properties.get('proj:bbox')
                asset_info["shape"] = asset.properties.get('proj:shape')
            elif hasattr(asset, 'title'):
                asset_info["description"] = asset.title or ""

            assets_info[asset_key] = asset_info

        logger.info(f"Retrieved detailed information for {len(assets_info)} assets from item '{item.id}'")
        return assets_info

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
    require_full_bbox_coverage: bool = False,
    max_cloud_cover: Optional[int] = None
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
        max_cloud_cover: Maximum cloud cover percentage (0-100). If None, no cloud cover filter is applied.

    Returns:
        List of STAC items.
    """
    try:
        client = MPCSTACClient()
        query = {}
        if path is not None:
            query["landsat:wrs_path"] = {"eq": path}
        if row is not None:
            query["landsat:wrs_row"] = {"eq": row}
        if mgrs_tile is not None:
            query["s2:mgrs_tile"] = {"eq": mgrs_tile}
        if max_cloud_cover is not None:
            query["eo:cloud_cover"] = {"lte": max_cloud_cover}
        kwargs = {}
        if query:
            kwargs["query"] = query
        items = client.search_items(collection, bbox, datetime_range, limit, **kwargs)
    except Exception as e:
        logger.error(f"Error searching collection '{collection}': {e}")
        return []

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


def get_mpc_item_assets(item_id: str, collection: str) -> List[str]:
    """
    Convenience function to get assets from a specific MPC item.

    Args:
        item_id: Item ID.
        collection: Collection ID.

    Returns:
        List of asset keys available in the item.
    """
    client = MPCSTACClient()
    item = client.get_item(item_id, collection)
    if item:
        return client.get_item_assets(item)
    return []


def get_mpc_collection_band_details(collection: str) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed band information for a collection using hardcoded data for known collections.

    This function provides hardcoded band information for Landsat and Sentinel-2 collections
    instead of dynamically retrieving from STAC API.

    Args:
        collection (str): Collection ID.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping band keys to detailed band information.
            Each band info dict contains:
            - 'name': Band name/key
            - 'description': Band description
            - 'wavelength': Center wavelength (if available)
            - 'resolution': Spatial resolution/GSD (if available)
            - 'common_name': Common name (if available)
    """
    # Hardcoded band information for Landsat collections
    landsat_bands = {
        'qa': {
            'name': 'qa',
            'description': 'Surface Temperature Quality Assessment Band',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Quality Assessment'
        },
        'ang': {
            'name': 'ang',
            'description': 'Angle Coefficients File',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Angle Coefficients'
        },
        'red': {
            'name': 'red',
            'description': 'Red Band (SR_B4) Surface Reflectance',
            'wavelength': 0.65,
            'resolution': 30.0,
            'common_name': 'Red'
        },
        'blue': {
            'name': 'blue',
            'description': 'Blue Band (SR_B2) Surface Reflectance',
            'wavelength': 0.48,
            'resolution': 30.0,
            'common_name': 'Blue'
        },
        'drad': {
            'name': 'drad',
            'description': 'Downwelled Radiance Band (ST_DRAD) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Downwelled Radiance'
        },
        'emis': {
            'name': 'emis',
            'description': 'Emissivity Band (ST_EMIS) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Emissivity'
        },
        'emsd': {
            'name': 'emsd',
            'description': 'Emissivity Standard Deviation Band (ST_EMSD) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Emissivity SD'
        },
        'trad': {
            'name': 'trad',
            'description': 'Thermal Radiance Band (ST_TRAD) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Thermal Radiance'
        },
        'urad': {
            'name': 'urad',
            'description': 'Upwelled Radiance Band (ST_URAD) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Upwelled Radiance'
        },
        'atran': {
            'name': 'atran',
            'description': 'Atmospheric Transmittance Band (ST_ATRAN) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Atmospheric Transmittance'
        },
        'cdist': {
            'name': 'cdist',
            'description': 'Cloud Distance Band (ST_CDIST) Surface Temperature Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Cloud Distance'
        },
        'green': {
            'name': 'green',
            'description': 'Green Band (SR_B3) Surface Reflectance',
            'wavelength': 0.56,
            'resolution': 30.0,
            'common_name': 'Green'
        },
        'nir08': {
            'name': 'nir08',
            'description': 'Near Infrared Band 0.8 (SR_B5) Surface Reflectance',
            'wavelength': 0.86,
            'resolution': 30.0,
            'common_name': 'NIR'
        },
        'lwir11': {
            'name': 'lwir11',
            'description': 'Surface Temperature Band (ST_B10) Surface Temperature',
            'wavelength': 10.9,
            'resolution': 100.0,
            'common_name': 'Thermal Infrared'
        },
        'swir16': {
            'name': 'swir16',
            'description': 'Short-wave Infrared Band 1.6 (SR_B6) Surface Reflectance',
            'wavelength': 1.61,
            'resolution': 30.0,
            'common_name': 'SWIR 1.6'
        },
        'swir22': {
            'name': 'swir22',
            'description': 'Short-wave Infrared Band 2.2 (SR_B7) Surface Reflectance',
            'wavelength': 2.20,
            'resolution': 30.0,
            'common_name': 'SWIR 2.2'
        },
        'coastal': {
            'name': 'coastal',
            'description': 'Coastal/Aerosol Band (SR_B1) Surface Reflectance',
            'wavelength': 0.44,
            'resolution': 30.0,
            'common_name': 'Coastal/Aerosol'
        },
        'mtl.txt': {
            'name': 'mtl.txt',
            'description': 'Product Metadata File (txt)',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Metadata TXT'
        },
        'mtl.xml': {
            'name': 'mtl.xml',
            'description': 'Product Metadata File (xml)',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Metadata XML'
        },
        'mtl.json': {
            'name': 'mtl.json',
            'description': 'Product Metadata File (json)',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Metadata JSON'
        },
        'qa_pixel': {
            'name': 'qa_pixel',
            'description': 'Pixel Quality Assessment Band (QA_PIXEL)',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Pixel QA'
        },
        'qa_radsat': {
            'name': 'qa_radsat',
            'description': 'Radiometric Saturation and Terrain Occlusion Quality Assessment Band (QA_RADSAT)',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Radiometric Saturation QA'
        },
        'qa_aerosol': {
            'name': 'qa_aerosol',
            'description': 'Aerosol Quality Assessment Band (SR_QA_AEROSOL) Surface Reflectance Product',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Aerosol QA'
        },
        'tilejson': {
            'name': 'tilejson',
            'description': 'TileJSON with default rendering',
            'wavelength': None,
            'resolution': None,
            'common_name': 'TileJSON'
        },
        'rendered_preview': {
            'name': 'rendered_preview',
            'description': 'Rendered preview',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Preview'
        }
    }

    # Hardcoded band information for Sentinel-2 collections
    sentinel2_bands = {
        'AOT': {
            'name': 'AOT',
            'description': 'Aerosol optical thickness',
            'wavelength': None,
            'resolution': 10.0,
            'common_name': 'Aerosol Optical Thickness'
        },
        'B01': {
            'name': 'B01',
            'description': 'Band 1 - Coastal aerosol - 60m',
            'wavelength': 0.44,
            'resolution': 60.0,
            'common_name': 'Coastal Aerosol'
        },
        'B02': {
            'name': 'B02',
            'description': 'Band 2 - Blue - 10m',
            'wavelength': 0.49,
            'resolution': 10.0,
            'common_name': 'Blue'
        },
        'B03': {
            'name': 'B03',
            'description': 'Band 3 - Green - 10m',
            'wavelength': 0.56,
            'resolution': 10.0,
            'common_name': 'Green'
        },
        'B04': {
            'name': 'B04',
            'description': 'Band 4 - Red - 10m',
            'wavelength': 0.66,
            'resolution': 10.0,
            'common_name': 'Red'
        },
        'B05': {
            'name': 'B05',
            'description': 'Band 5 - Vegetation red edge 1 - 20m',
            'wavelength': 0.70,
            'resolution': 20.0,
            'common_name': 'Red Edge 1'
        },
        'B06': {
            'name': 'B06',
            'description': 'Band 6 - Vegetation red edge 2 - 20m',
            'wavelength': 0.74,
            'resolution': 20.0,
            'common_name': 'Red Edge 2'
        },
        'B07': {
            'name': 'B07',
            'description': 'Band 7 - Vegetation red edge 3 - 20m',
            'wavelength': 0.78,
            'resolution': 20.0,
            'common_name': 'Red Edge 3'
        },
        'B08': {
            'name': 'B08',
            'description': 'Band 8 - NIR - 10m',
            'wavelength': 0.84,
            'resolution': 10.0,
            'common_name': 'NIR'
        },
        'B09': {
            'name': 'B09',
            'description': 'Band 9 - Water vapor - 60m',
            'wavelength': 0.94,
            'resolution': 60.0,
            'common_name': 'Water Vapor'
        },
        'B11': {
            'name': 'B11',
            'description': 'Band 11 - SWIR (1.6) - 20m',
            'wavelength': 1.61,
            'resolution': 20.0,
            'common_name': 'SWIR 1.6'
        },
        'B12': {
            'name': 'B12',
            'description': 'Band 12 - SWIR (2.2) - 20m',
            'wavelength': 2.19,
            'resolution': 20.0,
            'common_name': 'SWIR 2.2'
        },
        'B8A': {
            'name': 'B8A',
            'description': 'Band 8A - Vegetation red edge 4 - 20m',
            'wavelength': 0.86,
            'resolution': 20.0,
            'common_name': 'Red Edge 4'
        },
        'SCL': {
            'name': 'SCL',
            'description': 'Scene classification map',
            'wavelength': None,
            'resolution': 20.0,
            'common_name': 'Scene Classification'
        },
        'WVP': {
            'name': 'WVP',
            'description': 'Water vapour',
            'wavelength': None,
            'resolution': 10.0,
            'common_name': 'Water Vapour'
        },
        'visual': {
            'name': 'visual',
            'description': 'True color image',
            'wavelength': None,
            'resolution': 10.0,
            'common_name': 'True Color'
        },
        'safe-manifest': {
            'name': 'safe-manifest',
            'description': 'SAFE manifest',
            'wavelength': None,
            'resolution': None,
            'common_name': 'SAFE Manifest'
        },
        'granule-metadata': {
            'name': 'granule-metadata',
            'description': 'Granule metadata',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Granule Metadata'
        },
        'inspire-metadata': {
            'name': 'inspire-metadata',
            'description': 'INSPIRE metadata',
            'wavelength': None,
            'resolution': None,
            'common_name': 'INSPIRE Metadata'
        },
        'product-metadata': {
            'name': 'product-metadata',
            'description': 'Product metadata',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Product Metadata'
        },
        'datastrip-metadata': {
            'name': 'datastrip-metadata',
            'description': 'Datastrip metadata',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Datastrip Metadata'
        },
        'tilejson': {
            'name': 'tilejson',
            'description': 'TileJSON with default rendering',
            'wavelength': None,
            'resolution': None,
            'common_name': 'TileJSON'
        },
        'rendered_preview': {
            'name': 'rendered_preview',
            'description': 'Rendered preview',
            'wavelength': None,
            'resolution': None,
            'common_name': 'Preview'
        }
    }

    # Check if collection is Landsat or Sentinel-2
    collection_lower = collection.lower()
    if 'landsat' in collection_lower:
        logger.info(f"Using hardcoded Landsat band information for collection '{collection}'")
        return landsat_bands
    elif 'sentinel-2' in collection_lower or 'sentinel2' in collection_lower:
        logger.info(f"Using hardcoded Sentinel-2 band information for collection '{collection}'")
        return sentinel2_bands
    else:
        logger.warning(f"No hardcoded band information available for collection '{collection}' - returning empty dict")
        return {}


def get_mpc_collection_bands(collection: str) -> List[str]:
    """
    Get available bands for a collection with error handling and fallback mechanism.

    Args:
        collection: Collection ID.

    Returns:
        List of band names, or empty list if bands cannot be retrieved.
    """
    try:
        # Try the primary method using STAC client
        client = MPCSTACClient()
        collection_obj = client.client.get_collection(collection)
        
        # Try to get bands from the collection's summaries
        if hasattr(collection_obj, 'summaries') and collection_obj.summaries:
            bands_data = collection_obj.summaries.get_list("eo:bands")
            logger.info(f"bands_data for collection '{collection}': {bands_data}")
            if bands_data:
                # Extract band names from the data structure
                if bands_data and isinstance(bands_data[0], dict):
                    # If bands are dictionaries, extract the 'name' field or first string value
                    bands = []
                    for band in bands_data:
                        if isinstance(band, dict):
                            # Try common band name fields
                            band_name = band.get('name') or band.get('common_name') or band.get('center_wavelength')
                            if band_name:
                                bands.append(str(band_name))
                            else:
                                # Use first string value found
                                for key, value in band.items():
                                    if isinstance(value, str):
                                        bands.append(value)
                                        break
                        else:
                            bands.append(str(band))
                else:
                    # If bands are already strings
                    bands = [str(band) for band in bands_data]
                
                logger.info(f"Retrieved {len(bands)} bands for collection '{collection}' using STAC summaries")
                logger.info(f"Returning bands: {bands}")
                return bands
        
        # Try to get bands from the first item in the collection
        try:
            search = client.client.search(collections=[collection], limit=1)
            items = list(search.items())
            if items:
                first_item = items[0]
                if hasattr(first_item, 'assets'):
                    bands = list(first_item.assets.keys())
                    logger.info(f"Retrieved {len(bands)} bands for collection '{collection}' from first item assets")
                    logger.info(f"Returning bands: {bands}")
                    return bands
        except Exception as e:
            logger.warning(f"Could not get bands from first item: {e}")
        
        # Try pandas-based approach as fallback (as suggested in todo.txt)
        try:
            import pandas as pd
            catalog = client.client
            collection_obj = catalog.get_collection(collection)
            
            if hasattr(collection_obj, 'summaries') and collection_obj.summaries:
                bands_data = collection_obj.summaries.get_list("eo:bands")
                if bands_data:
                    bands_df = pd.DataFrame(bands_data)
                    if not bands_df.empty:
                        # Extract band names from the DataFrame
                        bands = []
                        for _, row in bands_df.iterrows():
                            if isinstance(row.iloc[0], dict):
                                # Handle dictionary rows
                                band_name = row.iloc[0].get('name') or row.iloc[0].get('common_name')
                                if band_name:
                                    bands.append(str(band_name))
                                else:
                                    # Use first string value
                                    for value in row.iloc[0].values():
                                        if isinstance(value, str):
                                            bands.append(value)
                                            break
                            else:
                                # Handle direct string values
                                bands.append(str(row.iloc[0]))
                        
                        logger.info(f"Retrieved {len(bands)} bands for collection '{collection}' using pandas fallback")
                        logger.info(f"Returning bands: {bands}")
                        return bands
        except ImportError:
            logger.warning("Pandas not available for fallback band retrieval")
        except Exception as e:
            logger.warning(f"Pandas fallback failed: {e}")
        
        # Return empty list instead of None to prevent UI collapse
        logger.warning(f"No bands found for collection '{collection}' - returning empty list to prevent UI collapse")
        logger.info(f"Returning bands: []")
        return []

    except Exception as e:
        logger.error(f"Error getting bands for collection '{collection}': {e}")
        # Return empty list instead of None to prevent UI collapse
        logger.info(f"Returning bands: []")
        return []

