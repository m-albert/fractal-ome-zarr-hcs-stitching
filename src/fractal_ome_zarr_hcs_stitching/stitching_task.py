"""This is the Python module for sitching FOVs from an OME-Zarr image."""

import logging
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.ngff.zarr_utils import ZarrGroupNotFoundError
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import get_single_image_ROI
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.tasks._zarr_utils import (
    _split_well_path_image_path,
    _update_well_metadata,
)
from multiview_stitcher import (
    fusion,
    misc_utils,
    msi_utils,
    param_utils,
    registration,
)
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from ome_zarr import writer
from ome_zarr.io import parse_url
from pydantic import validate_call

from fractal_ome_zarr_hcs_stitching.utils import (
    PreRegistrationPruningMethod,
    StitchingChannelInputModel,
    get_sim_from_multiscales,
    get_tiles_from_sim,
)

logger = logging.getLogger(__name__)


@validate_call
def stitching_task(
    *,
    zarr_url: str,
    channel: StitchingChannelInputModel,
    overwrite_input: bool = False,
    output_group_suffix: str = "_fused",
    registration_resolution_level: int = 0,
    registration_on_z_proj: bool = True,
    pre_registration_pruning_method: PreRegistrationPruningMethod = PreRegistrationPruningMethod.KEEPAXISALIGNED,  # noqa: E501
    registration_n_jobs: int = 16,
    fusion_n_jobs: int = 16,
) -> None:
    """Stitches FOVs from an OME-Zarr image.

    Performs registration and fusion of FOVs indicated
    in the FOV_ROI_table of the OME-Zarr image. Writes the
    fused image back to a "fused" group in the same Zarr array.

    Todo:
      - include and update output metadata / FOV ROI table
      - test 2D / 3D
      - optimize for large data
      - currently optimized for search first mode, need to implement
        registration pair finding for "grid" (?) mode

    Args:
        zarr_url: Absolute path to the OME-Zarr image.
        channel: Channel for registration; requires either
            `wavelength_id` (e.g. `A01_C01`) or `label` (e.g. `DAPI`), but not
            both.
        overwrite_input: Whether to override the original, not stitched image
            with the output of this task.
        output_group_suffix: Suffix of the new OME-Zarr image to write the
            fused image to. Note that OME-Zarr (at least up to v0.5) doesn't
            allow non-alphanumeric characters in group names (set to e.g.
            just 'fused' instead of '_fused' for compliance).
        registration_resolution_level: Resolution level to use for registration.
        registration_on_z_proj: Whether to perform registration on a maximum
            projection along z in case of 3D data.
        pre_registration_pruning_method: Method to use for selecting a subset
            of all overlapping tiles for pairwise registration. By default,
            only lower, upper, right and left neighbors are considered. Set
            this parameter to no_pruning if pairs of tiles which deviate
            from this pattern need to be registered.
        registration_n_jobs: Number of parallel pairwise registrations to run.
            Setting this is specifically useful for limiting memory usage.
            If the task runs out of memory during registration, reduce this number.
            Default is 4.
        fusion_n_jobs: Number of parallel jobs to use for fusion to zarr.
            Uses joblib for parallelization. It makes sense to set
            this to the number of cores available to the task. If the task
            runs out of memory during fusion, reduce this number.
            Default is 4.
    """
    logger.info(f"{zarr_url=}")

    # Parse and log several NGFF-image metadata attributes
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    logger.info(f"  Axes: {ngff_image_meta.axes_names}")
    logger.info(f"  Number of pyramid levels: {ngff_image_meta.num_levels}")
    logger.info(
        f"Linear coarsening factor for YX axes: {ngff_image_meta.coarsening_xy}"
    )
    logger.info(
        "Full-resolution ZYX pixel sizes (micrometer): "
        f"{ngff_image_meta.get_pixel_sizes_zyx(level=0)}"
    )
    logger.info(
        "  Coarsening-level-1 ZYX pixel sizes (micrometer): "
        f"{ngff_image_meta.get_pixel_sizes_zyx(level=1)}"
    )

    # Load FOV ROI table
    fov_roi_table = ad.read_zarr(Path(zarr_url) / "tables/FOV_ROI_table").to_df()

    # Set transform key for input tiles
    # (in OME-Zarr version <= 0.5 there's a single transform key for input tiles)
    input_transform_key = "fractal_input"

    #############
    # Registration
    ##############

    # Load FOVs for registration
    xim_well_reg = get_sim_from_multiscales(
        Path(zarr_url), resolution=registration_resolution_level
    )

    # Determine whether to perform registration on maximum projection in Z:
    # Two cases:
    # 1) User requested it via registration_on_z_proj=True
    # 2) The data is 2D (size in z == 1)
    reg_max_project_z = registration_on_z_proj or xim_well_reg.sizes["z"] == 1

    if reg_max_project_z:
        xim_well_reg = xim_well_reg.max("z")

    # Get individual tile images for registration
    # (in the input data and xim_well_reg, the FOVs are subregions of a single
    # array defined by fov_roi_table)
    msims_reg = get_tiles_from_sim(
        xim_well_reg, fov_roi_table, transform_key=input_transform_key
    )

    reg_spatial_dims = si_utils.get_spatial_dims_from_sim(
        xim_well_reg.squeeze(drop=True)
    )

    logger.info("Started registration")
    logger.info(f"Registration res level: {registration_resolution_level}")
    logger.info(f"Registration spatial dims: {reg_spatial_dims}")

    # Make sure the requested channel is available in the OME-Zarr image
    omero_channel = channel.get_omero_channel(zarr_url)
    if omero_channel is None:
        logger.info(
            f"Skipping stitching for {zarr_url} because {channel} is "
            "not available in that OME-Zarr image"
        )
        return

    # Perform the actual registration
    # Try-except block to catch NotEnoughOverlapError in case no overlapping
    # tiles are found for registration. If that happens, we skip registration
    # and directly proceed to fusion
    try:
        fusion_transform_key = "translation_registered"
        params = registration.register(
            msims_reg,
            transform_key=input_transform_key,
            new_transform_key=fusion_transform_key,
            reg_channel=omero_channel.label,
            registration_binning={dim: 1 for dim in reg_spatial_dims},
            pre_registration_pruning_method=pre_registration_pruning_method.get_pruning_method(),
            n_parallel_pairwise_regs=registration_n_jobs,
        )
        shifts = {
            ip: {
                dim: s
                for dim, s in zip(
                    reg_spatial_dims,
                    param_utils.translation_from_affine(p.sel(t=0).data),
                )
            }
            for ip, p in enumerate(params)

            if not np.allclose(p.sel(t=0).data, np.eye(len(reg_spatial_dims) + 1))
        }
        logger.info(f"Obtained shifts: {shifts}")
    except NotEnoughOverlapError:
        logger.warning(
            "Did not find overlapping tiles for stitching. Skipping registration."
        )
        fusion_transform_key = input_transform_key

    logger.info("Finished registration")

    ########
    # Fusion
    ########

    # If registration was performed at full resolution and without
    # maximum projection in Z, we can reuse the loaded data for fusion.
    # Otherwise, read full-resolution data for fusion
    if registration_resolution_level == 0 and not reg_max_project_z:
        xim_well = xim_well_reg
        msims_fusion = msims_reg
    else:
        xim_well = get_sim_from_multiscales(Path(zarr_url), resolution=0)
        msims_fusion = get_tiles_from_sim(
            xim_well, fov_roi_table, transform_key=input_transform_key
        )

    # Assign the registration parameters to the tiles to be fused
    # multiview-stitcher attaches the transformations obtained during registration
    # to the tile objects. However, if registration was performed on different
    # data (e.g. max projection in Z, or lower resolution), we need to
    # reattach and/or broadcast the obtained transforms to the tile data to be fused.
    for itile in range(len(msims_fusion)):

        # Get the affine transform from the registered tile used during registration
        affine = msi_utils.get_transform_from_msim(
            msims_reg[itile], fusion_transform_key
        )

        # If the registration was performed on a maximum projection in Z, we need to
        # broadcast the obtained affine parameters to 3D
        if reg_max_project_z:
            affine_3d = param_utils.identity_transform(
                ndim=3, t_coords=affine.coords["t"] if "t" in affine.dims else None
            )
            affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
            affine = affine_3d

        # Set the affine transform for the tile to be fused
        msi_utils.set_affine_transform(
            msims_fusion[itile], affine, fusion_transform_key
        )

    # Fusion input are single resolution spatial-images
    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion]
    
    # If "t" not in input_dims, remove the dimension from sims before
    # fusion to avoid writing t dim to zarr:
    # multiview-stitcher currently adds and/or requires a (at least dummy) t dimension
    sims = [si_utils.sim_sel_coords(sim, sel_dict={'t': 0}) for sim in sims]

    # Determine output zarr url
    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    output_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}{output_group_suffix}"
    logger.info(f"Output fused path: {output_zarr_url}")

    # Fuse directly to zarr at highest resolution (level 0)
    # This writes only the full-resolution data without building a large dask graph
    logger.info("Started fusion computation (writing directly to zarr)")
    logger.info(f"Using {fusion_n_jobs} parallel jobs.")

    fused = fusion.fuse(
        sims,
        transform_key=fusion_transform_key,
        output_chunksize={dim: sims[0].data.chunksize[idim]
                          for idim, dim in enumerate(sims[0].dims)},
        output_spacing=si_utils.get_spacing_from_sim(sims[0]),
        # fusion_func=fusion.max_fusion,
        output_zarr_url=f"{output_zarr_url}/0",
        zarr_options={
            "ome_zarr": False,  # Don't use multiview-stitcher to create OME-Zarr metadata
            "overwrite": True,
        },
        batch_options={
            "batch_func": misc_utils.process_batch_using_joblib,
            "n_batch": 1000, # num of chunks to schedule at once in joblib
            "batch_func_kwargs": {
                "n_jobs": fusion_n_jobs,
            },
        },
    )

    logger.info("Finished fusion at full resolution")
    logger.info("Started building resolution pyramid")

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    # Provide original chunksize to avoid "ValueError: Attempt to save array
    # to zarr with irregular chunking, please call `arr.rechunk(...)` first."
    build_pyramid(
        zarrurl=output_zarr_url,
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        chunksize=sims[0].data.chunksize,
        coarsening_xy=ngff_image_meta.coarsening_xy,
        open_array_kwargs={"write_empty_chunks": False, "fill_value": 0},
    )

    # Attach OME-Zarr metadata to the fused image
    store = parse_url(output_zarr_url, mode="w").store
    output_group = zarr.group(store=store)
    writer.write_multiscales_metadata(
        group=output_group,
        axes=ngff_image_meta.axes_names,
        datasets=[
            {
                "path": fractal_ds.path,
                "coordinateTransformations": [
                    {
                        "type": coordinateTransformation.type,
                        "scale": coordinateTransformation.scale,
                    }
                    for coordinateTransformation in fractal_ds.coordinateTransformations
                ],
            }
            for fractal_ds in ngff_image_meta.multiscales[0].datasets[
                : ngff_image_meta.num_levels
            ]
        ],
    )
    output_group.attrs["omero"] = ngff_image_meta.omero.model_dump()

    # Workaround: Manually add wavelength_id attr back to omero channel
    original_omero_attrs = zarr.open(zarr_url).attrs["omero"]["channels"]
    for i, omero_channel in enumerate(output_group.attrs["omero"]["channels"]):
        omero_channel["wavelength_id"] = original_omero_attrs[i]["wavelength_id"]
    output_attrs = output_group.attrs
    output_group.attrs["omero"] = dict(output_attrs["omero"])

    logger.info("Finished building resolution pyramid")

    # Add ROI table to the image
    ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pixels_ZYX = (
        ngff_image_meta.multiscales[0]
        .datasets[0]
        .coordinateTransformations[0]
        .scale[-3:]
    )
    image_ROI_table = get_single_image_ROI(fused.data.shape, pixels_ZYX=pixels_ZYX)
    write_table(
        output_group,
        "well_ROI_table",  # Could also be image_ROI_table
        image_ROI_table,
        overwrite=True,
        table_attrs={"type": "roi_table"},
    )

    ####################
    # Clean up Zarr file
    ####################
    if overwrite_input:
        logger.info("Replace original zarr image with the newly created Zarr image")
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(output_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=output_zarr_url, origin=zarr_url)]
        )
        # Update the metadata of the the well
        well_url, new_img_path = _split_well_path_image_path(output_zarr_url)
        try:
            _update_well_metadata(
                well_url=well_url,
                old_image_path=old_img_path,
                new_image_path=new_img_path,
            )
        except ZarrGroupNotFoundError:
            logger.debug(
                f"{zarr_url} is not in an HCS plate. No well metadata got updated"
            )
        except ValueError:
            logger.debug(
                f"Could not update well metadata, likely because "
                f" {output_zarr_url} was already listed there."
            )

        return image_list_updates

    logger.info("Done stitching.")


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=stitching_task)
