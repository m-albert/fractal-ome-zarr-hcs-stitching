"""
Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""
from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "fractal_ome_zarr_hcs_stitching"
    create_manifest(package=PACKAGE)
