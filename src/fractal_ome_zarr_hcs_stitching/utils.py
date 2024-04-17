from pathlib import Path

import dask.array as da
from spatial_image import to_spatial_image

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta

from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import msi_utils, param_utils


def get_sim_from_multiscales(multiscales_path, resolution):

    ngff_image_meta = load_NgffImageMeta(multiscales_path)
    axes = ngff_image_meta.axes_names
    spatial_dims = [dim for dim in axes if dim in ["z", "y", "x"]]
    scales = ngff_image_meta.pixel_sizes_zyx

    channel_names = [oc.label
        for oc in get_omero_channel_list(image_zarr_path=multiscales_path)]
    
    data = da.from_zarr(f"{multiscales_path / Path(str(resolution))}")

    sim = to_spatial_image(
        data,
        dims=axes,
        c_coords=channel_names,
        scale={dim: scales[resolution][idim]
                for idim, dim in enumerate(spatial_dims)},
        translation={dim: 0 for dim in spatial_dims},
    )

    return sim


def get_tiles_from_sim(xim_well, fov_roi_table):

    input_spatial_dims = [dim for dim in xim_well.dims if dim in ["z", "y", "x"]]
    msims = []
    for i, row in fov_roi_table.iterrows():

        origin = {dim: row[f'{dim}_micrometer'] for dim in input_spatial_dims}
        extent = {dim: row[f'len_{dim}_micrometer'] for dim in input_spatial_dims}

        origin_original = {dim: row[f'{dim}_micrometer_original'] if dim != 'z' else 0 for dim in input_spatial_dims}
        extent_original = {dim: row[f'len_{dim}_micrometer'] for dim in input_spatial_dims}

        tile = xim_well.sel({dim: slice(origin[dim], origin[dim] + extent[dim])
                            for dim in input_spatial_dims})
        
        tile = tile.squeeze(drop=True)

        tile_spatial_dims = [dim for dim in tile.dims if dim in ["z", "y", "x"]]
            
        sim = si_utils.get_sim_from_array(
            tile.data,
            dims=tile.dims,
            c_coords=xim_well.coords['c'].data,
            scale=si_utils.get_spacing_from_sim(tile),
            translation=origin_original,
            transform_key='fractal_original',
        )

        msim = msi_utils.get_msim_from_sim(
            sim,
            scale_factors=[]
            )

        msims.append(msim)

    return msims
