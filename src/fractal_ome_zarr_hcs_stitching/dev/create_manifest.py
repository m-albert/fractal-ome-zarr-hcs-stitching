"""Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "fractal_ome_zarr_hcs_stitching"
    AUTHORS = "Marvin Albert"
    create_manifest(
        package=PACKAGE,
        authors=AUTHORS,
        custom_pydantic_models=[
            (
                "fractal_ome_zarr_hcs_stitching",
                "utils.py",
                "StitchingChannelInputModel",
            )
        ],
    )
