"""Generate JSON schemas for task arguments afresh, and write them
to the package manifest.
"""

from fractal_tasks_core.dev.create_manifest import create_manifest

if __name__ == "__main__":
    PACKAGE = "fractal_ome_zarr_hcs_stitching"
    AUTHORS = "Marvin Albert, Joel Luethi, Nicole Repina"
    docs_link = "https://github.com/m-albert/fractal-ome-zarr-hcs-stitching"
    create_manifest(
        package=PACKAGE,
        authors=AUTHORS,
        docs_link=docs_link,
        custom_pydantic_models=[
            (
                "fractal_ome_zarr_hcs_stitching",
                "utils.py",
                "StitchingChannelInputModel",
            )
        ],
    )
