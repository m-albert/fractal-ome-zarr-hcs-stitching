"""
This is the Python module for my_task
"""

import logging
from typing import Any
from pathlib import Path

import zarr
import dask.array as da

from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid

@validate_arguments
def thresholding_task(
    *,
    zarr_url: str,
    another_parameter: int = 1,
) -> None:
    """
    Short description of thresholding_task.

    Long description of thresholding_task.

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        another_parameter: An arbitrary parameter.
    """

    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

    # Parse and log several NGFF-image metadata attributes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    logging.info(f"  Axes: {ngff_image_meta.axes_names}")
    logging.info(f"  Number of pyramid levels: {ngff_image_meta.num_levels}")
    logging.info(f"  Linear coarsening factor for YX axes: {ngff_image_meta.coarsening_xy}")
    logging.info(f"  Full-resolution ZYX pixel sizes (micrometer):    {ngff_image_meta.get_pixel_sizes_zyx(level=0)}")
    logging.info(f"  Coarsening-level-1 ZYX pixel sizes (micrometer): {ngff_image_meta.get_pixel_sizes_zyx(level=1)}")

    # Load the highest-resolution multiscale array through dask.array
    array_czyx = da.from_zarr(f"{zarr_url}/0")
    logging.info(f"{array_czyx=}")

    # Set values below 100 to 0
    array_max = array_czyx.max().compute()
    array_min = array_czyx.min().compute()
    logging.info(f"Pre thresholding:  {array_min=}, {array_max=}")
    array_czyx[array_czyx < 99] = 99
    array_czyx[array_czyx > 1000] = 1000
    array_max = array_czyx.max().compute()
    array_min = array_czyx.min().compute()
    logging.info(f"Post thresholding: {array_min=}, {array_max=}")

    # Write the processed array back to the same full-resolution Zarr array
    array_czyx.to_zarr(f"{zarr_url}/0", overwrite=True)

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarr_url,
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        coarsening_xy=ngff_image_meta.coarsening_xy,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=thresholding_task)