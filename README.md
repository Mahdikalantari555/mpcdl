# mpcdl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for downloading remote sensed data easily from Microsoft Planetary Computer (MPC).

## Installation

Install mpcdl directly from GitHub:

```bash
pip install git+https://github.com/Mahdikalantari555/mpcdl.git
```

Install on Conda environment – recommended:

This method creates an isolated Conda environment and installs mpcdl from source while resolving geospatial dependencies cleanly.

1. Clone the repository
git clone https://github.com/Mahdikalantari555/mpcdl.git
cd mpcdl

2. Create a Conda environment
conda create -f environment.yml -y
conda activate mpcdl

Verification
python -c "import mpcdl; print('mpcdl installed successfully')"


## Usage

Here's a basic example of searching and downloading Sentinel-2 data:

```python
import mpcdl

# Search for Sentinel-2 data
items = mpcdl.search_mpc_collection(
    collection="sentinel-2-l2a",
    bbox=[-122.5, 37.7, -122.3, 37.8],
    datetime_range="2023-06-01/2023-06-30",
    limit=5
)

print(f"Found {len(items)} items")

# Download assets from the first item
if items:
    assets = mpcdl.download_stac_assets(
        items[0].assets,
        output_dir="./data",
        asset_keys=["B04", "B03", "B02"]
    )
    print("Downloaded RGB bands")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.