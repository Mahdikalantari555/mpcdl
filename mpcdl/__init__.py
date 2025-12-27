"""
mpcdl Package

A package for downloading and processing STAC data from Microsoft Planetary Computer.

"""

__version__ = "0.1.0"

# Import main modules
from . import download, load, preprocess, stac, utils

# Import key classes and functions for easy access
from .stac import MPCSTACClient, search_mpc_collection, get_mpc_item, get_item_assets
from .download import download_file, download_stac_assets, download_batch, download_mpc_dataset, download_stac_assets_clipped
from .load import load_raster, load_vector, load_geojson_from_url, load_raster_from_url, load_stac_item_assets
from .preprocess import clip_raster, reproject_raster, resample_raster, normalize_raster, stack_rasters, preprocess_pipeline
from .utils import setup_logging, load_environment_variables, get_env_var, ensure_directory, log_package_info

# Log package information on import
log_package_info()