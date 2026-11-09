"""Fractal multiview stitcher utils."""

import logging
from enum import Enum
from typing import Optional

import pandas as pd
from multiview_stitcher import spatial_image_utils as si_utils
from ngio import OmeZarrContainer
from ngio.utils import NgioValueError
from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


def get_sim_from_ome_zarr_container(
    ome_zarr_container: OmeZarrContainer,
    resolution: int = 0,
    transform_key: str = "fractal_input",
    as_dask: bool = False,
):
    """Get a spatial image representing a given resolution level of an image.

    Parameters
    ----------
    ome_zarr_container : ngio.OmeZarrContainer
        OME-Zarr image container.
    resolution : int, optional
        Resolution level index, by default 0
    transform_key : str, optional
        Transform key to attach the (identity) input transform to.
    as_dask : bool, optional
        Back the spatial image by a dask array instead of the zarr array.
        Needed for reductions such as `.max("z")`, which are computed eagerly
        (i.e. load the whole array) on zarr-backed spatial images.
        By default False.

    Returns:
    -------
    spatial-image (multiview-stitcher flavor)
    """
    image = ome_zarr_container.get_image(
        path=ome_zarr_container.level_paths[resolution]
    )
    axes = list(image.axes)
    spatial_dims = [dim for dim in axes if dim in ["z", "y", "x"]]
    pixel_size = image.pixel_size

    return si_utils.get_sim_from_array(
        # By default pass the zarr array directly: multiview-stitcher then
        # reads only the regions it needs, and passes compact references to
        # the zarr array (instead of dask graphs) to parallel fusion workers
        image.get_as_dask() if as_dask else image.zarr_array,
        dims=axes,
        # multiview-stitcher identifies channels by their `c` coordinate,
        # so channel labels are used (they must be unique within the image)
        c_coords=image.channel_labels,
        scale={dim: getattr(pixel_size, dim) for dim in spatial_dims},
        translation={dim: 0 for dim in spatial_dims},
        transform_key=transform_key,
    )


def get_fov_sims_from_well_sim(
    xim_well,
    fov_roi_table: pd.DataFrame,
    transform_key: str = "fractal_input",
):
    """Get FOV spatial images from a well spatial image
    and a (Fractal) FOV ROI table.

    Parameters
    ----------
    xim_well : spatial_image.SpatialImage
        Array representing the well.
    fov_roi_table : pd.DataFrame
        Table with the FOV ROIs.

    Returns:
    -------
    list of spatial-images (multiview-stitcher flavor)
    """
    input_spatial_dims = [dim for dim in xim_well.dims if dim in ["z", "y", "x"]]
    fov_sims = []
    for _, row in fov_roi_table.iterrows():
        origin = {dim: row[f"{dim}_micrometer"] for dim in input_spatial_dims}
        extent = {dim: row[f"len_{dim}_micrometer"] for dim in input_spatial_dims}

        origin_original = {
            dim: row[f"{dim}_micrometer_original"] if dim != "z" else 0
            for dim in input_spatial_dims
        }

        tile = xim_well.sel(
            {
                dim: slice(origin[dim], origin[dim] + extent[dim] - 1e-6)
                for dim in input_spatial_dims
            }
        )

        sim = si_utils.get_sim_from_array(
            # Pass the DataArray, not `tile.data`: for zarr-backed tiles,
            # `.data` would load the tile into memory
            tile,
            dims=tile.dims,
            c_coords=xim_well.coords["c"].data,
            scale=si_utils.get_spacing_from_sim(tile),
            translation=origin_original,
            transform_key=transform_key,
        )

        fov_sims.append(sim)

    return fov_sims


class StitchingChannelInputModel(BaseModel):
    """Channel input for stitching."""

    # Same fields and validation as fractal-tasks-core's (v1) ChannelInputModel,
    # so that existing workflows keep working.
    # Attribute docstrings are used as field descriptions in the manifest.
    model_config = ConfigDict(use_attribute_docstrings=True)

    wavelength_id: Optional[str] = None
    """
    Unique ID for the channel wavelength, e.g. `A01_C01`.
    Can only be specified if label is not set.
    """

    label: Optional[str] = None
    """
    Name of the channel. Can only be specified if wavelength_id is not set.
    """

    @model_validator(mode="after")
    def mutually_exclusive_channel_attributes(self):
        """Check that exactly one of `label` or `wavelength_id` is set."""
        # Setting both would also make ngio's `get_channel_idx` raise, which
        # `get_channel_label` would misreport as "channel not found"
        if (self.wavelength_id is None) == (self.label is None):
            raise ValueError(
                "One and only one of `wavelength_id` and `label` must be set "
                f"(given: {self.wavelength_id=}, {self.label=})."
            )
        return self

    def get_channel_label(
        self, ome_zarr_container: OmeZarrContainer
    ) -> Optional[str]:
        """Get the label of the selected channel, or None if it is not found."""
        try:
            channel_idx = ome_zarr_container.get_channel_idx(
                channel_label=self.label, wavelength_id=self.wavelength_id
            )
        except NgioValueError as e:
            # Missing channel: the caller skips this image instead of failing
            logger.warning(
                f"Channel with wavelength_id: {self.wavelength_id} "
                f"and label: {self.label} not found, exit from the task.\n"
                f"Original error: {e!s}"
            )
            return None
        # Return the label (also when selecting by wavelength_id), as this is
        # what multiview-stitcher uses to select the registration channel
        return ome_zarr_container.channel_labels[channel_idx]


class PreRegistrationPruningMethod(Enum):
    """PreRegistrationPruningMethod Enum class

    Attributes:
        NOPRUNING: All overlapping tiles are used for registration.
        KEEPAXISALIGNED: Use only orthogonal tile pairs for registration.
            This excludes diagonal tile pairs and can lead to more robust
            registration results when tiles are not positioned irregularily.
        SHORTESTPATHSOVERLAPWEIGHTED: Only the tile pairs required to connect
            all tiles to an automatically determined reference tile are used
            for registration. Tile pairs with high overlaps are preferred.
    """

    NOPRUNING = "no_pruning"
    KEEPAXISALIGNED = "keep_axis_aligned"
    SHORTESTPATHSOVERLAPWEIGHTED = "shortest_paths_overlap_weighted"

    def get_pruning_method(self):
        """Get the pruning method to use

        Complex mapping for no pruning, see
        https://github.com/fractal-analytics-platform/fractal-web/issues/714
        for context. NOPRUNING should return a None, not the string.
        """
        return None if self == PreRegistrationPruningMethod.NOPRUNING else self.value
