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
from multiview_stitcher import fusion, msi_utils, param_utils, registration
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher.mv_graph import NotEnoughOverlapError
from ome_zarr import writer
from ome_zarr.io import parse_url
from pydantic.v1.decorator import validate_arguments

from fractal_ome_zarr_hcs_stitching.utils import (
    get_sim_from_multiscales,
    get_tiles_from_sim,
)

logger = logging.getLogger(__name__)


@validate_arguments
def stitching_task(
    *,
    zarr_url: str,
    registration_channel_label: str = "DAPI",
    overwrite_input: bool = False,
    output_group_suffix: str = "fused",
    registration_resolution_level: int = 0,
    registration_on_z_proj: bool = True,
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
        registration_channel_label: Label of the channel to use for registration.
        overwrite_input: Whether to override the original, not stitched image
            with the output of this task.
        output_group_suffix: Suffix of the new OME-Zarr image to write the
            fused image to.
        registration_resolution_level: Resolution level to use for registration.
        registration_on_z_proj: Whether to perform registration on a maximum
            projection along z in case of 3D data.
    """
    # Use the first of input_paths
    logging.info(f"{zarr_url=}")

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

    fov_roi_table = ad.read_zarr(Path(zarr_url) / "tables/FOV_ROI_table").to_df()
    input_transform_key = "fractal_input"

    #############
    # Registration
    ##############

    # Load FOVs for registration
    xim_well_reg = get_sim_from_multiscales(
        Path(zarr_url), resolution=registration_resolution_level
    )  # could also be lower resolution

    input_spatial_dims = si_utils.get_spatial_dims_from_sim(
        xim_well_reg.squeeze(drop=True)
    )

    # determine whether to perform registration on maximum projection in Z
    reg_max_project_z = registration_on_z_proj and ("z" in input_spatial_dims)

    if reg_max_project_z:
        xim_well_reg = xim_well_reg.max("z")

    msims_reg = get_tiles_from_sim(
        xim_well_reg, fov_roi_table, transform_key=input_transform_key
    )

    reg_spatial_dims = si_utils.get_spatial_dims_from_sim(
        xim_well_reg.squeeze(drop=True)
    )

    logger.info("Started registration")
    logger.info(f"Registration channel: {registration_channel_label}")
    logger.info(f"Registration res level: {registration_resolution_level}")
    logger.info(f"Registration spatial dims: {reg_spatial_dims}")

    reg_channel_index = (
        xim_well_reg.coords["c"].data.tolist().index(registration_channel_label)
    )
    try:
        fusion_transform_key = "translation_registered"
        params = registration.register(
            msims_reg,
            transform_key=input_transform_key,
            new_transform_key=fusion_transform_key,
            reg_channel_index=reg_channel_index,
            registration_binning={dim: 1 for dim in reg_spatial_dims},
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

    if registration_resolution_level == 0 and not reg_max_project_z:
        xim_well = xim_well_reg
        msims_fusion = msims_reg
    else:
        # Load the full-resolution image for fusion
        xim_well = get_sim_from_multiscales(Path(zarr_url), resolution=0)
        msims_fusion = get_tiles_from_sim(
            xim_well, fov_roi_table, transform_key=input_transform_key
        )

    # assign the registration parameters to the tiles to be fused
    for itile in range(len(msims_fusion)):
        affine = msi_utils.get_transform_from_msim(
            msims_reg[itile], fusion_transform_key
        )

        # if the registration was performed on a maximum projection in Z, we need to
        # broadcast the obtained affine parameters to 3D
        if reg_max_project_z:
            affine_3d = param_utils.identity_transform(
                ndim=3, t_coords=affine.coords["t"] if "t" in affine.dims else None
            )
            affine_3d.loc[{pdim: affine.coords[pdim] for pdim in affine.dims}] = affine
            affine = affine_3d

        msi_utils.set_affine_transform(
            msims_fusion[itile], affine, fusion_transform_key
        )

    sims = [msi_utils.get_sim_from_msim(msim) for msim in msims_fusion]

    logger.info(f"Started fusion using transform key {fusion_transform_key}")
    fused = fusion.fuse(
        sims,
        transform_key=fusion_transform_key,
        output_chunksize=xim_well.data.chunksize[-1],
        output_spacing=si_utils.get_spacing_from_sim(sims[0]),
        # fusion_func=fusion.max_fusion,
    )

    fused = fused.sel(t=0, drop=True)

    if "z" not in fused.dims:
        fused = fused.expand_dims("z", xim_well.dims.index("z"))

    # get the dask array from the fused sim
    fused_da = fused.sel({"c": fused.coords["c"].values}).data

    # rechunk the fused array to match the original array
    fused_da = fused_da.rechunk(xim_well.data.chunksize)

    well_url, old_img_path = _split_well_path_image_path(zarr_url)
    output_zarr_url = f"{well_url}/{zarr_url.split('/')[-1]}_{output_group_suffix}"
    logger.info(f"Output fused path: {output_zarr_url}")

    # Write the fused array back to the same full-resolution Zarr array
    fused_da = fused_da.to_zarr(
        f"{output_zarr_url}/0",
        overwrite=True,
        dimension_separator="/",
        return_stored=True,
        compute=True,
    )

    logger.info("Finished fusion")

    logger.info("Started building resolution pyramid")

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    # Provide original chunksize to avoid "ValueError: Attempt to save array
    # to zarr with irregular chunking, please call `arr.rechunk(...)` first."
    build_pyramid(
        zarrurl=output_zarr_url,
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        chunksize=xim_well.data.chunksize,
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
        metadata=dict(
            omero=dict(
                channels=[channel.dict() for channel in ngff_image_meta.omero.channels]
            )
        ),
    )
    # Workaround: Manually add wavelength_id attr back to omero channel
    original_omero_attrs = zarr.open(zarr_url).attrs["omero"]["channels"]
    for i, omero_channel in enumerate(output_group.attrs["omero"]["channels"]):
        omero_channel["wavelength_id"] = original_omero_attrs[i]["wavelength_id"]

    logger.info("Finished building resolution pyramid")

    # Add ROI table to the image
    ngff_image_meta.get_pixel_sizes_zyx(level=0)
    pixels_ZYX = (
        ngff_image_meta.multiscales[0]
        .datasets[0]
        .coordinateTransformations[0]
        .scale[-3:]
    )
    image_ROI_table = get_single_image_ROI(fused_da.shape, pixels_ZYX=pixels_ZYX)
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

    logger.info("Done stitching")


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=stitching_task)
