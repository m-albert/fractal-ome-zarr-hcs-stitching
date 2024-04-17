"""
This is the Python module for my_task
"""

import logging
from typing import Any
from pathlib import Path

import zarr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar

import anndata as ad

from pydantic.decorator import validate_arguments

from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid

from fractal_ome_zarr_hcs_stitching.utils import (
    get_sim_from_multiscales,
    get_tiles_from_sim,
)

from multiview_stitcher import (
    registration,
    fusion,
    msi_utils,
    param_utils
)
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from multiview_stitcher import spatial_image_utils as si_utils


@validate_arguments
def stitching_task(
    *,
    zarr_url: str,
    registration_channel_label: str = "DAPI",
) -> None:
    """
    Stitches FOVs from an OME-Zarr image.

    Performs registration and fusion of FOVs indicated
    in the FOV_ROI_table of the OME-Zarr image. Writes the
    fused image back to a "fused" group in the same Zarr array.

    TODO:
      - include and update output metadata / FOV ROI table
      - test 2D / 3D
      - how to determine num_levels for build_pyramid? fails

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        registration_channel_label: Label of the channel to use for registration.
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

    # Load FOVs for registration
    xim_well = get_sim_from_multiscales(Path(zarr_url), 0) # could also be lower resolution
    fov_roi_table = ad.read_zarr(Path(zarr_url) / "tables/FOV_ROI_table").to_df()
    msims = get_tiles_from_sim(xim_well, fov_roi_table)

    logging.info(f"Started registration")
    reg_channel_index = xim_well.coords['c']\
        .data.tolist().index(registration_channel_label)
    try:
        with ProgressBar():
            params = registration.register(
                msims[:],
                transform_key='fractal_original',
                reg_channel_index=reg_channel_index,
                new_transform_key='translation_registered',
                registration_binning={'y': 2, 'x': 2},
            )
    except NotEnoughOverlapError:
        logging.error(f"Not enough overlap for stitching.")
        return
    logging.info(f"Finished registration")

    reg_spatial_dims = [dim for dim in msi_utils.get_sim_from_msim(msims[0]).dims
        if dim in ['z', 'y', 'x']]
    shifts = {ip:
                {dim: s for dim, s in zip(
                    reg_spatial_dims,
                    param_utils.translation_from_affine(p.sel(t=0).data)
                    )
                }
                    for ip, p in enumerate(params)
                    if not np.allclose(p[0].data, np.eye(len(reg_spatial_dims) + 1))}
    logging.info(f"Obtained shifts: {shifts}")

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    logging.info(f"Started fusion")
    # with dask.config.set(**{'array.slicing.split_large_chunks': False}):
    fused = fusion.fuse(
        sims[:],
        transform_key='translation_registered',
        output_chunksize=xim_well.data.chunksize[-1],
        output_spacing=si_utils.get_spacing_from_sim(sims[0]),
        )

    fused = fused.sel(t=0, drop=True)

    if 'z' not in fused.dims:
        fused = fused.expand_dims('z', xim_well.dims.index('z'))

    # get the dask array from the fused sim
    fused_da = fused.data#[..., :2000, :2000]

    # # Homogenize chunks (?)
    # fused_da = da.pad(
    #     fused_da,
    #     [(0, int(cs - (s % cs)))
    #         for cs, s in zip(fused_da.chunksize, fused_da.shape)])
    # fused_da = fused_da.rechunk(xim_well.data.chunksize)

    # Write the fused array back to the same full-resolution Zarr array
    fused_da.to_zarr(f"{zarr_url}/fused/0", overwrite=True)

    logging.info(f"Finished fusion")

    logging.info(f"Started building resolution pyramid")

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    # How to determine number of levels and coarsening factors?
    # `build_pyramid`` fails with certain combinations of shape and num_levels
    build_pyramid(
        zarrurl=f"{zarr_url}/fused",
        overwrite=True,
        num_levels=3,
        coarsening_xy=ngff_image_meta.coarsening_xy,
    )

    logging.info(f"Finished building resolution pyramid")


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=stitching_task)