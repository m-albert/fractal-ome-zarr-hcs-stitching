"""This is the Python module for sitching FOVs from an OME-Zarr image."""

import logging
import os
import shutil

import numpy as np
from multiview_stitcher import (
    fusion,
    misc_utils,
    msi_utils,
    param_utils,
    registration,
)
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from ngio import open_ome_zarr_container, open_ome_zarr_well
from ngio.utils import NgioFileNotFoundError
from pydantic import validate_call

from fractal_ome_zarr_hcs_stitching.utils import (
    PreRegistrationPruningMethod,
    StitchingChannelInputModel,
    get_fov_sims_from_well_sim,
    get_sim_from_ome_zarr_container,
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
    """Stitches fields of view (FOV) in an OME-Zarr image.

    Registers and fuses FOVs specified in the Fractal FOV_ROI_table.
    Writes the fused image back to a group within the same OME-Zarr image.

    Features:
    - Input can be 2D or 3D
    - Registration is performed
      - on a specified resolution level
      - on a specified channel
      - optionally on a maximum projection for 3D data
    - Supports large data with low memory consumption
    - Supports regular and irregular FOV layouts (e.g. FOVs from "Search First" mode)
    - Parallelized registration and fusion

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
            of all overlapping FOVs/tiles for pairwise registration. By default,
            only lower, upper, right and left neighbors are considered. Set
            this parameter to no_pruning if pairs of tiles which deviate
            from this pattern need to be registered.
        registration_n_jobs: Number of parallel pairwise registrations to run.
            Setting this is specifically useful for limiting memory usage.
            If the task runs out of memory during registration, reduce this number.
            Default is 16.
        fusion_n_jobs: Number of parallel jobs to use for fusion to zarr.
            Uses joblib for parallelization. It makes sense to set
            this to the number of cores available to the task. If the task
            runs out of memory during fusion, reduce this number.
            Default is 16.
    """
    logger.info(f"{zarr_url=}")

    # Open the OME-Zarr image and log several metadata attributes
    ome_zarr_container = open_ome_zarr_container(zarr_url)
    input_image = ome_zarr_container.get_image()
    logger.info(f"  Axes: {input_image.axes}")
    logger.info(f"  Pyramid levels: {ome_zarr_container.level_paths}")
    logger.info(f"  Full-resolution pixel size: {input_image.pixel_size}")

    # Make sure the requested channel is available in the OME-Zarr image.
    # Checked first, so that images without it are skipped before any work.
    reg_channel_label = channel.get_channel_label(ome_zarr_container)
    if reg_channel_label is None:
        logger.info(
            f"Skipping stitching for {zarr_url} because {channel} is "
            "not available in that OME-Zarr image"
        )
        return

    # Load FOV ROI table
    fov_roi_table = ome_zarr_container.get_roi_table("FOV_ROI_table").dataframe

    # Set transform key for input tiles
    # (in OME-Zarr version <= 0.5 there's a single transform key for input tiles)
    input_transform_key = "fractal_input"

    #############
    # Registration
    ##############

    # Load FOVs for registration
    xim_well_reg = get_sim_from_ome_zarr_container(
        ome_zarr_container,
        resolution=registration_resolution_level,
        transform_key=input_transform_key,
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
    sims_reg = get_fov_sims_from_well_sim(
        xim_well_reg, fov_roi_table, transform_key=input_transform_key
    )

    # Registration expects multiscale spatial images as input
    msims_reg = [msi_utils.get_msim_from_sim(sim, scale_factors=[]) for sim in sims_reg]

    reg_spatial_dims = si_utils.get_spatial_dims_from_sim(
        xim_well_reg.squeeze(drop=True)
    )

    logger.info("Started registration")
    logger.info(f"Registration res level: {registration_resolution_level}")
    logger.info(f"Registration spatial dims: {reg_spatial_dims}")

    # Perform the actual registration
    # Try-except block to catch NotEnoughOverlapError in case no overlapping
    # FOVs are found for registration. If that happens, we skip registration
    # and directly proceed to fusion
    try:
        fusion_transform_key = "translation_registered"
        params = registration.register(
            msims_reg,
            transform_key=input_transform_key,
            new_transform_key=fusion_transform_key,
            # Only this channel is used for registration; the obtained
            # transforms are applied to all channels during fusion
            reg_channel=reg_channel_label,
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
            "Did not find overlapping FOVs for stitching. Skipping registration."
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
        sims_fusion = [msi_utils.get_sim_from_msim(msim) for msim in msims_reg]
    else:
        xim_well = get_sim_from_ome_zarr_container(
            ome_zarr_container, resolution=0, transform_key=input_transform_key
        )
        sims_fusion = get_fov_sims_from_well_sim(
            xim_well, fov_roi_table, transform_key=input_transform_key
        )

    # Assign the registration parameters to the FOVs to be fused
    # multiview-stitcher attaches the transformations obtained during registration
    # to the tile objects. However, if registration was performed on different
    # data (e.g. max projection in Z, or lower resolution), we need to
    # reattach and/or broadcast the obtained transforms to the tile data to be fused.
    for itile in range(len(sims_fusion)):

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
        si_utils.set_sim_affine(
            sims_fusion[itile], affine, fusion_transform_key
        )
    # If "t" not in input_dims, remove the dimension from sims before
    # fusion to avoid writing t dim to zarr:
    # multiview-stitcher currently adds and/or requires a (at least dummy) t dimension
    sims_fusion = [si_utils.sim_sel_coords(sim, sel_dict={'t': 0}) for sim in sims_fusion]

    # Determine output zarr url
    well_url, old_img_path = zarr_url.rstrip("/").rsplit("/", 1)
    output_zarr_url = f"{well_url}/{old_img_path}{output_group_suffix}"
    logger.info(f"Output fused path: {output_zarr_url}")

    # Determine the output geometry (union of all registered tiles), so that
    # the output image can be created before fusing into it
    output_stack_properties = fusion.process_output_stack_properties(
        sims_fusion,
        output_spacing=si_utils.get_spacing_from_sim(sims_fusion[0]),
        transform_key=fusion_transform_key,
    )
    # Spatial sizes come from the fusion geometry, others (e.g. c) from the input
    output_shape = [
        int(output_stack_properties["shape"][axis])
        if axis in output_stack_properties["shape"]
        else input_image.dimensions.get(axis)
        for axis in input_image.axes
    ]
    logger.info(f"Output shape: {output_shape}")

    # Create the output image with the same metadata (channels, pixel sizes,
    # number of pyramid levels, NGFF version, chunks) as the input image.
    # Channel metadata (labels, wavelength_ids, colors, windows) is copied over.
    output_container = ome_zarr_container.derive_image(
        store=output_zarr_url,
        shape=output_shape,
        chunks=input_image.chunks,
        overwrite=True,
        copy_labels=False,
        copy_tables=False,
    )
    output_image = output_container.get_image()

    # Fuse directly into the full-resolution (level 0) array of the output
    # image. Fusion is performed chunk by chunk, without building a large dask
    # graph. Chunks are aligned with the output array chunks so that parallel
    # writes never touch the same chunk.
    logger.info("Started fusion computation (writing directly to zarr)")
    logger.info(f"Using {fusion_n_jobs} parallel jobs.")

    block_fusion = fusion.prepare_block_fusion(
        f"{output_zarr_url}/{output_container.level_paths[0]}",
        fuse_kwargs={
            "images": sims_fusion,
            "transform_key": fusion_transform_key,
            "output_stack_properties": output_stack_properties,
            "output_chunksize": {
                dim: output_image.chunks[output_image.axes.index(dim)]
                for dim in sims_fusion[0].dims
                if dim in output_image.axes
            },
        },
        # The array was already created by ngio (derive_image) with the
        # right zarr format, compression and metadata: write into it
        create_output=False,
    )
    for batch in misc_utils.ndindex_batches(
        block_fusion["nblocks"],
        1000,  # num of chunks to schedule at once in joblib
    ):
        misc_utils.process_batch_using_joblib(
            block_fusion["func"], batch, n_jobs=fusion_n_jobs
        )

    logger.info("Finished fusion at full resolution")
    logger.info("Started building resolution pyramid")

    # Build the coarser pyramid levels from the on-disk full-resolution data
    # (block averaging, as previously done by fractal-tasks-core)
    output_image.consolidate(mode="coarsen")

    logger.info("Finished building resolution pyramid")

    # Add ROI table covering the whole fused image
    output_container.add_table(
        "well_ROI_table",  # Could also be image_ROI_table
        output_container.build_image_roi_table("image"),
        overwrite=True,
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
        new_img_path = output_zarr_url.rsplit("/", 1)[1]
        try:
            open_ome_zarr_well(well_url).atomic_add_image(image_path=new_img_path)
        except NgioFileNotFoundError:
            logger.debug(
                f"{zarr_url} is not in an HCS plate. No well metadata got updated"
            )
        except ValueError:  # NgioValueError is a ValueError
            logger.debug(
                f"Could not update well metadata, likely because "
                f" {output_zarr_url} was already listed there."
            )

        return image_list_updates

    logger.info("Done stitching.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=stitching_task)
