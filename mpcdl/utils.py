"""
Utils module

This module contains utility functions and helpers for the package.

"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dotenv import load_dotenv
import yaml
import json
from datetime import datetime
import importlib.metadata
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer

logger = logging.getLogger(__name__)


def setup_logging(
    level: Union[str, int] = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None
) -> None:
    """
    Setup logging configuration for the package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) as string or int.
        format_string: Custom format string for log messages.
        log_file: Optional file path to save logs.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Handle both string and int level inputs
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper())
    else:
        numeric_level = level

    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(),
            *(logging.FileHandler(log_file) for _ in [log_file] if log_file)
        ]
    )

    logger.info(f"Logging setup complete with level {level}")


def load_environment_variables(env_file: Union[str, Path] = ".env") -> None:
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file.
    """
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f"Environment file {env_path} not found")


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable with optional default.

    Args:
        key: Environment variable key.
        default: Default value if key not found.

    Returns:
        Environment variable value or default.
    """
    value = os.getenv(key, default)
    if value is None:
        logger.warning(f"Environment variable '{key}' not found")
    return value


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_package_version() -> str:
    """
    Get the version of the pympc package.

    Returns:
        Package version string.
    """
    try:
        version = importlib.metadata.version("pympc")
        return version
    except importlib.metadata.PackageNotFoundError:
        # Fallback to reading from __init__.py
        init_file = Path(__file__).parent / "__init__.py"
        if init_file.exists():
            with open(init_file, "r") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"\'')


def log_package_info() -> None:
    """
    Log package version and dependency information.
    """
    version = get_package_version()
    logger.info(f"pympc version: {version}")

    # Log key dependency versions
    try:
        import rasterio
        logger.info(f"rasterio version: {rasterio.__version__}")
    except ImportError:
        logger.warning("rasterio not available")

    try:
        import geopandas
        logger.info(f"geopandas version: {geopandas.__version__}")
    except ImportError:
        logger.warning("geopandas not available")

    try:
        import pystac
        logger.info(f"pystac version: {pystac.__version__}")
    except ImportError:
        logger.warning("pystac not available")


def read_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration data.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def write_yaml_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Write configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the YAML file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Saved configuration to {config_path}")


def read_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Dictionary containing configuration data.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def write_json_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Write configuration to a JSON file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the JSON file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved configuration to {config_path}")


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get the size of a file in megabytes.

    Args:
        file_path: Path to the file.

    Returns:
        File size in MB.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def validate_bbox(bbox: list) -> bool:
    """
    Validate a bounding box.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat].

    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False

    min_lon, min_lat, max_lon, max_lat = bbox

    if not all(isinstance(coord, (int, float)) for coord in bbox):
        return False

    if min_lon >= max_lon or min_lat >= max_lat:
        return False

    if not (-180 <= min_lon <= 180) or not (-180 <= max_lon <= 180):
        return False

    if not (-90 <= min_lat <= 90) or not (-90 <= max_lat <= 90):
        return False

    return True


def timestamp_filename(prefix: str = "", suffix: str = "", extension: str = "") -> str:
    """
    Generate a timestamped filename.

    Args:
        prefix: Prefix for the filename.
        suffix: Suffix for the filename.
        extension: File extension (without dot).

    Returns:
        Timestamped filename string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [part for part in [prefix, timestamp, suffix] if part]
    filename = "_".join(parts)

    if extension:
        filename += f".{extension}"

    return filename


def reproject_bbox_to_wgs84(bbox: List[float], from_crs: Union[str, int]) -> List[float]:
    """
    Reproject a bounding box from any CRS to WGS84.

    Args:
        bbox: Bounding box as [min_x, min_y, max_x, max_y] in the source CRS.
        from_crs: Source coordinate reference system (EPSG code, WKT, or PROJ string).

    Returns:
        Bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84.
    """
    try:
        min_x, min_y, max_x, max_y = bbox
        
        # Create polygon from bbox
        polygon = Polygon([
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ])
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=from_crs)
        
        # Reproject to WGS84
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        
        # Extract bounding box
        bounds = gdf_wgs84.total_bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]]
        
    except Exception as e:
        logger.error(f"Failed to reproject bbox from CRS {from_crs}: {e}")
        raise


def get_layer_bbox_wgs84(layer) -> Optional[List[float]]:
    """
    Get the bounding box of a QGIS layer in WGS84 coordinates.

    Args:
        layer: QGIS map layer object.

    Returns:
        Bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84, or None if error.
    """
    try:
        # Get layer extent
        extent = layer.extent()
        
        # Create bbox from extent
        bbox = [extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()]
        
        # Get layer CRS
        layer_crs = layer.crs()
        
        if not layer_crs or layer_crs.isGeographic():
            # Already in geographic coordinates, no reprojection needed
            return bbox
        
        # Reproject to WGS84
        return reproject_bbox_to_wgs84(bbox, layer_crs.authid())
        
    except Exception as e:
        logger.error(f"Failed to get layer bbox in WGS84: {e}")
        return None


def get_map_extent_bbox_wgs84(extent, map_crs) -> List[float]:
    """
    Get map extent bounding box in WGS84 coordinates.

    Args:
        extent: QGIS map extent object.
        map_crs: QGIS coordinate reference system object.

    Returns:
        Bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84.
    """
    try:
        # Create bbox from extent
        bbox = [extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()]
        
        if not map_crs or map_crs.isGeographic():
            # Already in geographic coordinates, no reprojection needed
            return bbox
        
        # Reproject to WGS84
        return reproject_bbox_to_wgs84(bbox, map_crs.authid())
        
    except Exception as e:
        logger.error(f"Failed to reproject map extent to WGS84: {e}")
        raise


def get_geodataframe_bbox_wgs84(gdf: gpd.GeoDataFrame) -> List[float]:
    """
    Get bounding box of a GeoDataFrame in WGS84 coordinates.

    Args:
        gdf: GeoDataFrame object.

    Returns:
        Bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84.
    """
    try:
        # Ensure GeoDataFrame is in WGS84
        if gdf.crs != "EPSG:4326":
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
        else:
            gdf_wgs84 = gdf
        
        # Get bounds
        bounds = gdf_wgs84.total_bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]]
        
    except Exception as e:
        logger.error(f"Failed to get GeoDataFrame bbox in WGS84: {e}")
        raise


def extract_layer_geometry_wgs84(layer) -> Optional[gpd.GeoDataFrame]:
    """
    Extract actual vector geometry from a QGIS layer and reproject to WGS84.

    Args:
        layer: QGIS vector layer object.

    Returns:
        GeoDataFrame with geometry in WGS84 coordinates, or None if error.
    """
    try:
        from qgis.core import Qgis
        
        # Check if layer is a vector layer
        if layer.type() != layer.VectorLayer:
            logger.warning(f"Layer '{layer.name()}' is not a vector layer")
            return None
        
        # Get layer CRS
        layer_crs = layer.crs()
        if not layer_crs or layer_crs.isGeographic():
            logger.info(f"Layer '{layer.name()}' is already in geographic coordinates")
        
        # Extract features and create GeoDataFrame
        features = []
        geometries = []
        
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom and not geom.isEmpty():
                # Convert QGIS geometry to Shapely
                shapely_geom = geom.constGet().clone()
                if shapely_geom:
                    geometries.append(shapely_geom)
        
        if not geometries:
            logger.warning(f"No valid geometries found in layer '{layer.name()}'")
            return None
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=layer_crs.authid() if layer_crs else "EPSG:4326")
        
        # Reproject to WGS84 if necessary
        if layer_crs and not layer_crs.isGeographic():
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            logger.info(f"Reprojected layer '{layer.name()}' geometry to WGS84")
            return gdf_wgs84
        else:
            return gdf
            
    except Exception as e:
        logger.error(f"Failed to extract geometry from layer '{layer.name()}': {e}")
        return None


def merge_geometries_to_multipolygon(gdf: gpd.GeoDataFrame) -> Optional[Polygon]:
    """
    Merge multiple geometries in a GeoDataFrame into a single MultiPolygon.

    Args:
        gdf: GeoDataFrame with polygon geometries.

    Returns:
        Shapely MultiPolygon or Polygon, or None if error.
    """
    try:
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import unary_union
        
        if gdf.empty or gdf.geometry.is_empty.all():
            logger.warning("GeoDataFrame is empty or contains no valid geometries")
            return None
        
        # Ensure all geometries are polygons
        polygon_geoms = []
        for geom in gdf.geometry:
            if geom.geom_type in ['Polygon', 'MultiPolygon']:
                polygon_geoms.append(geom)
            elif geom.geom_type in ['LineString', 'MultiLineString']:
                # Try to buffer lines to create polygons
                buffered = geom.buffer(0.0001)  # Small buffer to create area
                if buffered.geom_type in ['Polygon', 'MultiPolygon']:
                    polygon_geoms.append(buffered)
            elif geom.geom_type in ['Point', 'MultiPoint']:
                # Buffer points to create small polygons
                buffered = geom.buffer(0.0001)
                if buffered.geom_type in ['Polygon', 'MultiPolygon']:
                    polygon_geoms.append(buffered)
        
        if not polygon_geoms:
            logger.warning("No valid polygon geometries found for merging")
            return None
        
        # Merge geometries using unary_union
        merged_geom = unary_union(polygon_geoms)
        
        logger.info(f"Merged {len(polygon_geoms)} geometries into {merged_geom.geom_type}")
        return merged_geom
        
    except Exception as e:
        logger.error(f"Failed to merge geometries: {e}")
        return None


def validate_geometry_for_clipping(geometry) -> bool:
    """
    Validate that geometry is suitable for clipping operations.

    Args:
        geometry: Shapely geometry object.

    Returns:
        True if geometry is valid for clipping, False otherwise.
    """
    try:
        if geometry is None:
            return False
        
        # Check if geometry is empty
        if hasattr(geometry, 'is_empty') and geometry.is_empty:
            logger.warning("Geometry is empty")
            return False
        
        # Check if geometry is valid
        if hasattr(geometry, 'is_valid') and not geometry.is_valid:
            logger.warning(f"Geometry is invalid: {geometry}")
            return False
        
        # Check geometry type
        valid_types = ['Polygon', 'MultiPolygon', 'LineString', 'MultiLineString', 'Point', 'MultiPoint']
        if hasattr(geometry, 'geom_type') and geometry.geom_type not in valid_types:
            logger.warning(f"Geometry type '{geometry.geom_type}' not suitable for clipping")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating geometry: {e}")
        return False


def create_mtl_json(
    item: 'pystac.Item',
    downloaded_files: Dict[str, Path],
    output_dir: Union[str, Path]
) -> bool:
    """
    Create MTL.json metadata file from STAC item properties.
    
    This function creates a comprehensive metadata JSON file containing:
    - Basic item info (ID, datetime, collection)
    - Geometry and bounding box
    - All STAC properties (cloud cover, sun angles, etc.)
    - List of downloaded files and their paths
    - Collection-specific metadata fields
    
    Args:
        item: STAC item
        downloaded_files: Dict mapping asset keys to downloaded file paths
        output_dir: Output directory for the MTL.json file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    from pathlib import Path
    import logging
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    try:
        output_dir = Path(output_dir)
        
        # Build comprehensive MTL data
        mtl_data = {
            "_metadata": {
                "created": datetime.now().isoformat(),
                "source": "MPC Downloader Plugin",
                "item_id": item.id,
                "collection": item.collection_id
            },
            "product_info": {
                "item_id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "cloud_cover": item.properties.get('eo:cloud_cover'),
                "bbox": item.bbox,
                "geometry": item.geometry
            }
        }
        
        # Add Landsat-specific fields if available
        if item.properties.get('landsat:wrs_path'):
            mtl_data["product_info"]["path"] = item.properties.get('landsat:wrs_path')
            mtl_data["product_info"]["row"] = item.properties.get('landsat:wrs_row')
            mtl_data["product_info"]["utm_zone"] = item.properties.get('utm:zone')
            mtl_data["product_info"]["epsg"] = item.properties.get('proj:epsg')
        
        # Add sun angles
        mtl_data["sun_angles"] = {
            "elevation": item.properties.get("view:sun_elevation"),
            "azimuth": item.properties.get("view:sun_azimuth")
        }
        
        # Add all STAC properties
        mtl_data["stac_properties"] = item.properties
        
        # Add downloaded files info
        mtl_data["downloaded_files"] = {
            str(k): str(v) for k, v in downloaded_files.items()
        }
        
        # Add list of downloaded asset keys
        mtl_data["downloaded_bands"] = list(downloaded_files.keys())
        
        # Add asset info (all available assets from the item)
        if hasattr(item, 'assets'):
            mtl_data["available_assets"] = list(item.assets.keys())
        
        # Write to file
        mtl_path = output_dir / "MTL.json"
        with open(mtl_path, 'w', encoding='utf-8') as f:
            json.dump(mtl_data, f, indent=2, default=str)
        
        logger.info(f"Created MTL.json: {mtl_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create MTL.json: {e}")
        return False


def create_full_mtl_json(
    item: 'pystac.Item',
    downloaded_files: Dict[str, Path],
    output_dir: Union[str, Path]
) -> bool:
    """
    Create a comprehensive MTL.json metadata file with full scene information.
    
    This function creates a complete metadata JSON file containing:
    - Basic item info: item_id, datetime, cloud_cover, collection
    - Geometry and bounding box
    - Path/row for Landsat, or other collection-specific identifiers
    - Sun angles (elevation, azimuth)
    - All STAC properties
    - List of all available assets
    - List of downloaded files
    - A comprehensive "METADATA" section with PRODUCT_METADATA, SENSOR_PARAMETERS,
      SUN_PARAMETERS, and other collection-specific fields
    
    Args:
        item: STAC item
        downloaded_files: Dict mapping asset keys to downloaded file paths
        output_dir: Output directory for the MTL.json file
        
    Returns:
        True if successful, False otherwise
    """
    import json
    from pathlib import Path
    import logging
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    def safe_get(props: Dict, key: str, default: Any = None) -> Any:
        """Safely get a value from properties, handling missing keys gracefully."""
        return props.get(key, default) if isinstance(props, dict) else default
    
    def extract_date_from_datetime(dt_str: Optional[str]) -> Optional[str]:
        """Extract date part (YYYY-MM-DD) from datetime string."""
        if not dt_str:
            return None
        try:
            # Try parsing as ISO format
            if 'T' in dt_str:
                return dt_str.split('T')[0]
            return dt_str
        except Exception:
            return None
    
    try:
        output_dir = Path(output_dir)
        properties = item.properties
        collection_id = item.collection_id or ""
        
        # Determine if this is a Landsat collection
        is_landsat = 'landsat' in collection_id.lower()
        
        # Build basic info section
        basic_info = {
            "item_id": item.id,
            "datetime": item.datetime.isoformat() if item.datetime else None,
            "date": extract_date_from_datetime(item.properties.get('datetime')),
            "cloud_cover": safe_get(properties, 'eo:cloud_cover'),
            "collection": collection_id
        }
        
        # Geometry and bbox
        geometry_info = {
            "bbox": item.bbox,
            "geometry": item.geometry
        }
        
        # Sun angles
        sun_angles = {
            "sun_elevation": safe_get(properties, 'view:sun_elevation'),
            "sun_azimuth": safe_get(properties, 'view:sun_azimuth')
        }
        
        # Path/Row for Landsat
        path_row = {}
        if is_landsat:
            path_row = {
                "wrs_path": safe_get(properties, 'landsat:wrs_path'),
                "wrs_row": safe_get(properties, 'landsat:wrs_row'),
                "utm_zone": safe_get(properties, 'utm:zone'),
                "epsg": safe_get(properties, 'proj:epsg')
            }
        
        # Build comprehensive METADATA section
        metadata = {
            "_note": "This MTL.json file was created from STAC properties by MPC Downloader Plugin"
        }
        
        # PRODUCT_METADATA section
        product_metadata = {
            "LANDSAT_PRODUCT_ID": item.id,
            "ACQUISITION_DATE": extract_date_from_datetime(safe_get(properties, 'datetime')),
            "CLOUD_COVER": safe_get(properties, 'eo:cloud_cover'),
            "WRS Path": safe_get(properties, 'landsat:wrs_path'),
            "WRS Row": safe_get(properties, 'landsat:wrs_row'),
            "STATION": safe_get(properties, 'station'),
            "SENSOR_ID": safe_get(properties, 'landsat:sensor_id'),
            "COLLECTION_NUMBER": safe_get(properties, 'landsat:collection_number'),
            "COLLECTION_CATEGORY": safe_get(properties, 'landsat:collection_category'),
            "CATEGORY": safe_get(properties, 'landsat:collection_category'),
            "CORRECTION_LEVEL": safe_get(properties, 'landsat:correction'),
            "DATA_TYPE": safe_get(properties, 'landsat:data_type')
        }
        
        # Remove None values from product_metadata
        product_metadata = {k: v for k, v in product_metadata.items() if v is not None}
        metadata["PRODUCT_METADATA"] = product_metadata
        
        # SENSOR_PARAMETERS section
        sensor_parameters = {
            "SENSOR_ID": safe_get(properties, 'sensor_id') or safe_get(properties, 'landsat:sensor_id'),
            "SPACECRAFT_ID": safe_get(properties, 'platform') or safe_get(properties, 'mission'),
            "INSTRUMENTS": safe_get(properties, 'instruments'),
            "FLATFORM": safe_get(properties, 'platform'),
            "MISSION": safe_get(properties, 'mission'),
            "SATELLITE": safe_get(properties, 'platform')
        }
        
        # Remove None values from sensor_parameters
        sensor_parameters = {k: v for k, v in sensor_parameters.items() if v is not None}
        metadata["SENSOR_PARAMETERS"] = sensor_parameters
        
        # SUN_PARAMETERS section
        sun_parameters = {
            "SUN_ELEVATION": safe_get(properties, 'view:sun_elevation'),
            "SUN_AZIMUTH": safe_get(properties, 'view:sun_azimuth')
        }
        
        # Remove None values from sun_parameters
        sun_parameters = {k: v for k, v in sun_parameters.items() if v is not None}
        metadata["SUN_PARAMETERS"] = sun_parameters
        
        # IMAGE_PROPERTIES section for non-Landsat collections
        if not is_landsat:
            image_properties = {
                "CLOUD_COVER": safe_get(properties, 'eo:cloud_cover'),
                "CLOUD_SHADOW_PERCENTAGE": safe_get(properties, 'eo:cloud_shadow_percentage'),
                "SNOW_ICE_PERCENTAGE": safe_get(properties, 'eo:snow_ice_percentage'),
                "CIRRUS_PERCENTAGE": safe_get(properties, 'eo:cirrus_percentage'),
                "NODATA_PIXEL_PERCENTAGE": safe_get(properties, 'eo:nodata_pixel_percentage'),
                "SATURATED_PIXEL_PERCENTAGE": safe_get(properties, 'eo:saturated_pixel_percentage')
            }
            image_properties = {k: v for k, v in image_properties.items() if v is not None}
            metadata["IMAGE_PROPERTIES"] = image_properties
        
        # Build the complete MTL data structure
        mtl_data = {
            "_metadata": {
                "created": datetime.now().isoformat(),
                "source": "MPC Downloader Plugin",
                "version": "1.0.0",
                "item_id": item.id,
                "collection": collection_id,
                "note": "Comprehensive MTL.json created from STAC properties"
            },
            "BASIC_INFO": basic_info,
            "GEOMETRY": geometry_info,
            "SUN_ANGLES": sun_angles
        }
        
        # Add path/row info for Landsat
        if is_landsat and path_row:
            mtl_data["PATH_ROW"] = path_row
        
        # Add METADATA section
        mtl_data["METADATA"] = metadata
        
        # Add all STAC properties
        mtl_data["STAC_PROPERTIES"] = properties
        
        # Add available assets
        if hasattr(item, 'assets'):
            mtl_data["AVAILABLE_ASSETS"] = list(item.assets.keys())
        
        # Add downloaded files info
        if downloaded_files:
            mtl_data["DOWNLOADED_FILES"] = {
                str(k): str(v) for k, v in downloaded_files.items()
            }
            mtl_data["DOWNLOADED_BANDS"] = list(downloaded_files.keys())
        
        # Add processing information if available
        processing_info = {
            "processing_level": safe_get(properties, 'processing:level'),
            "product_level": safe_get(properties, 'product_level'),
            "product_uri": safe_get(properties, 'product_uri'),
            "grid_spatial": safe_get(properties, 'grid:spatial_reference'),
            "proj": safe_get(properties, 'proj')
        }
        processing_info = {k: v for k, v in processing_info.items() if v is not None}
        if processing_info:
            mtl_data["PROCESSING_INFO"] = processing_info
        
        # Add temporal information
        temporal_info = {
            "start_datetime": item.datetime.isoformat() if item.datetime else None,
            "end_datetime": safe_get(properties, 'datetime:end'),
            "duration": safe_get(properties, 'duration')
        }
        temporal_info = {k: v for k, v in temporal_info.items() if v is not None}
        if temporal_info:
            mtl_data["TEMPORAL_INFO"] = temporal_info
        
        # Write to file
        mtl_path = output_dir / "MTL.json"
        with open(mtl_path, 'w', encoding='utf-8') as f:
            json.dump(mtl_data, f, indent=2, default=str)
        
        logger.info(f"Created comprehensive MTL.json: {mtl_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create comprehensive MTL.json: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_collection_metadata_assets(collection: str) -> list:
    """
    Get collection-specific metadata assets that should always be downloaded.
    
    Args:
        collection: The STAC collection ID (e.g., 'landsat-c2-l2', 'sentinel-2-l2a')
        
    Returns:
        List of metadata asset keys to download
    """
    # Collection-specific metadata assets
    metadata_assets = {
        'landsat-c2-l2': [
            'mtl.txt', 'mtl.xml', 'mtl.json', 'ang', 'qa_radsat', 'qa_aerosol'
        ],
        'sentinel-2-l2a': [
            'safe-manifest', 'product-metadata', 'granule-metadata', 
            'inspire-metadata', 'datastrip-metadata'
        ]
    }
    
    return metadata_assets.get(collection, [])


def geometry_to_geojson(geometry) -> Optional[Dict]:
    """
    Convert Shapely geometry to GeoJSON format.

    Args:
        geometry: Shapely geometry object.

    Returns:
        GeoJSON dictionary or None if error.
    """
    try:
        if not validate_geometry_for_clipping(geometry):
            return None
        
        # Convert to GeoJSON
        geojson = geometry.__geo_interface__
        return geojson
        
    except Exception as e:
        logger.error(f"Failed to convert geometry to GeoJSON: {e}")
        return None


def get_layer_geometry_wgs84(layer) -> Optional[Dict]:
    """
    Get layer geometry as GeoJSON in WGS84 coordinates.

    Args:
        layer: QGIS vector layer object.

    Returns:
        GeoJSON dictionary or None if error.
    """
    try:
        # Extract geometry from layer
        gdf = extract_layer_geometry_wgs84(layer)
        if gdf is None:
            return None
        
        # Merge geometries if multiple features exist
        if len(gdf) > 1:
            merged_geom = merge_geometries_to_multipolygon(gdf)
            if merged_geom is None:
                return None
            geometry = merged_geom
        else:
            geometry = gdf.geometry.iloc[0]
        
        # Validate and convert to GeoJSON
        if not validate_geometry_for_clipping(geometry):
            return None
        
        geojson = geometry_to_geojson(geometry)
        if geojson:
            logger.info(f"Successfully extracted geometry from layer '{layer.name()}'")
        
        return geojson
        
    except Exception as e:
        logger.error(f"Failed to get layer geometry as GeoJSON: {e}")
        return None