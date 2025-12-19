"""
Utils module

This module contains utility functions and helpers for the package.

"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv
import yaml
import json
from datetime import datetime
import importlib.metadata

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