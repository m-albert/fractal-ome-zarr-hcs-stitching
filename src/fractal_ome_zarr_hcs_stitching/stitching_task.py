"""
This is the Python module for sitching FOVs from an OME-Zarr image.
"""

import logging
from typing import Any
from pathlib import Path

import zarr
from ome_zarr import writer
from ome_zarr.io import parse_url
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
    output_group_name: str = "fused",
    registration_channel_label: str = "DAPI",
    registration_binning_xy: int = 1,
    registration_binning_z: int = 1,
) -> None:
    """
    Stitches FOVs from an OME-Zarr image.

    Performs registration and fusion of FOVs indicated
    in the FOV_ROI_table of the OME-Zarr image. Writes the
    fused image back to a "fused" group in the same Zarr array.

    TODO:
      - include and update output metadata / FOV ROI table
      - test 2D / 3D
      - optimize for large data
      - figure out how to best determine num_levels for build_pyramid
      - currently optimized for search first mode, need to implement
        registration pair finding for "grid" (?) mode

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        output_group_name: Name of the group to write the fused image to.
        registration_channel_label: Label of the channel to use for registration.
        registration_binning_xy: Binning factor for XY axes during registration.
        registration_binning_z: Binning factor for Z axis during registration (if present).
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

    reg_spatial_dims = [dim for dim in msi_utils.get_sim_from_msim(msims[0]).dims
        if dim in ['z', 'y', 'x']]

    #############
    # Registration
    ##############

    logging.info(f"Started registration")
    logging.info(f"Registration channel: {registration_channel_label}")
    logging.info(f"Registration binning XY: {registration_binning_xy}")
    logging.info(f"Registration binning Z: {registration_binning_z}")
    logging.info(f"Registration spatial dims: {reg_spatial_dims}")

    reg_channel_index = xim_well.coords['c']\
        .data.tolist().index(registration_channel_label)
    try:
        fusion_transform_key = 'translation_registered'
        params = registration.register(
            msims[:],
            transform_key='fractal_original',
            reg_channel_index=reg_channel_index,
            new_transform_key=fusion_transform_key,
            registration_binning={
                'y': registration_binning_xy,
                'x': registration_binning_xy,
                } if 'z' not in msi_utils.get_sim_from_msim(msims[0]).dims else {
                'z': registration_binning_z,
                'y': registration_binning_xy,
                'x': registration_binning_xy,
                },
        )
    except NotEnoughOverlapError:
        logging.warning(f"Not enough overlap for stitching.")
        fusion_transform_key = 'fractal_original'

    logging.info(f"Finished registration")

    shifts = {ip:
                {dim: s for dim, s in zip(
                    reg_spatial_dims,
                    param_utils.translation_from_affine(p.sel(t=0).data)
                    )
                }
                    for ip, p in enumerate(params)
                    if not np.allclose(p.sel(t=0).data, np.eye(len(reg_spatial_dims) + 1))}
    logging.info(f"Obtained shifts: {shifts}")

    ########
    # Fusion
    ########

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims]

    logging.info(f"Started fusion using transform key {fusion_transform_key}")
    fused = fusion.fuse(
        sims[:],
        transform_key=fusion_transform_key,
        output_chunksize=xim_well.data.chunksize[-1],
        output_spacing=si_utils.get_spacing_from_sim(sims[0]),
        # fusion_func=fusion.max_fusion,
        )

    fused = fused.sel(t=0, drop=True)

    if 'z' not in fused.dims:
        fused = fused.expand_dims('z', xim_well.dims.index('z'))

    # get the dask array from the fused sim
    fused_da = fused.data

    # rechunk the fused array to match the original array
    fused_da = fused_da.rechunk(xim_well.data.chunksize)

    output_zarr_url = f"{zarr_url}/{output_group_name}"
    logging.info(f"Output fused path: {output_zarr_url}")

    # Write the fused array back to the same full-resolution Zarr array
    fused_da = fused_da.to_zarr(
        f"{output_zarr_url}/0",
        overwrite=True,
        dimension_separator='/',
        return_stored=True,
        compute=True)

    logging.info(f"Finished fusion")

    logging.info(f"Started building resolution pyramid")

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    # How to determine number of levels and coarsening factors?
    # `build_pyramid`` fails with certain combinations of shape and num_levels
    output_num_levels = 4
    build_pyramid(
        zarrurl=output_zarr_url,
        overwrite=True,
        num_levels=output_num_levels,
        coarsening_xy=ngff_image_meta.coarsening_xy,
    )

    # attach metadata to the fused image
    store = parse_url(output_zarr_url, mode="w").store
    output_group = zarr.group(store=store)
    writer.write_multiscales_metadata(
        group=output_group,
        axes=ngff_image_meta.axes_names,
        datasets=[
            {
                'path': fractal_ds.path,
                'coordinateTransformations':
                    [
                        {
                            "type": coordinateTransformation.type,
                            "scale": coordinateTransformation.scale,
                        }
                    for coordinateTransformation in fractal_ds.coordinateTransformations
                    ]
            }
            for fractal_ds in ngff_image_meta.multiscales[0].datasets[:output_num_levels]
        ]
    )

    logging.info(f"Finished building resolution pyramid")
    logging.info(f"Done stitching")


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=stitching_task)