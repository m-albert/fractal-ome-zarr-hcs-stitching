"""
Fractal task(s) for registering and fusing OME-Zarr HCS
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-ome-zarr-hcs-stitching")
except PackageNotFoundError:
    __version__ = "uninstalled"