import pytest
import zarr

from fractal_ome_zarr_hcs_stitching.stitching_task import stitching_task
from fractal_ome_zarr_hcs_stitching.utils import StitchingChannelInputModel


def test_stitching_3d_search_first(
    search_first_ome_zarr_3d,
    registration_resolution_level=1,
    registration_on_z_proj=False,
):
    image_list_updates = stitching_task(
        zarr_url=search_first_ome_zarr_3d,
        channel=StitchingChannelInputModel(wavelength_id="A04_C01"),
        registration_resolution_level=registration_resolution_level,
        registration_on_z_proj=registration_on_z_proj,
    )
    expected_image_list_updates = {
        "image_list_updates": [
            {
                "zarr_url": f"{search_first_ome_zarr_3d}_fused",
                "origin": search_first_ome_zarr_3d,
            }
        ]
    }
    assert image_list_updates == expected_image_list_updates

    # This stitching runs, but is not very well fused due to interpolation
    # artefacts on the test data. Thus, checking the exact dimensions or other
    # outputs is not very meaningful here


@pytest.mark.parametrize(
    "pre_registration_pruning_method",
    ["keep_axis_aligned", "shortest_paths_overlap_weighted", "no_pruning"],
)
def test_stitching_3d_on_mip_search_first(
    pre_registration_pruning_method,
    search_first_ome_zarr_3d,
    registration_resolution_level=1,
    registration_on_z_proj=True,
):
    image_list_updates = stitching_task(
        zarr_url=search_first_ome_zarr_3d,
        channel=StitchingChannelInputModel(wavelength_id="A04_C01"),
        registration_resolution_level=registration_resolution_level,
        registration_on_z_proj=registration_on_z_proj,
        pre_registration_pruning_method=pre_registration_pruning_method,
    )
    expected_image_list_updates = {
        "image_list_updates": [
            {
                "zarr_url": f"{search_first_ome_zarr_3d}_fused",
                "origin": search_first_ome_zarr_3d,
            }
        ]
    }
    assert image_list_updates == expected_image_list_updates

    # Validate expected shape of the Zarr based on what was produced in
    # earlier tests that produces good fusion
    expected_shapes = {
        "keep_axis_aligned": (2, 6, 4383, 14580),
        "shortest_paths_overlap_weighted": (2, 6, 4383, 14580),
        "no_pruning": (2, 6, 4379, 14564),
    }
    with zarr.open(f"{search_first_ome_zarr_3d}_fused", mode="r") as zarr_group:
        assert zarr_group[0].shape == expected_shapes[pre_registration_pruning_method]


@pytest.mark.parametrize(
    "registration_resolution_level",
    [res_level for res_level in [0, 1]],
)
def test_stitching_2d_search_first(
    search_first_ome_zarr_2d,
    registration_resolution_level,
):
    image_list_updates = stitching_task(
        zarr_url=search_first_ome_zarr_2d,
        channel=StitchingChannelInputModel(wavelength_id="A04_C01"),
        registration_resolution_level=registration_resolution_level,
    )
    expected_image_list_updates = {
        "image_list_updates": [
            {
                "zarr_url": f"{search_first_ome_zarr_2d}_fused",
                "origin": search_first_ome_zarr_2d,
            }
        ]
    }
    assert image_list_updates == expected_image_list_updates
    # Validate expected shape of the Zarr based on what was produced in
    # earlier tests that produces good fusion
    expected_shapes = [
        (2, 1, 4392, 14580),
        (2, 1, 4391, 14580),
    ]
    with zarr.open(f"{search_first_ome_zarr_2d}_fused", mode="r") as zarr_group:
        assert zarr_group[0].shape == expected_shapes[registration_resolution_level]
        # Ensure the omero metadata is as expected (see #21):
        assert "metadata" not in zarr_group.attrs["multiscales"][0]
        assert zarr_group.attrs["omero"]["channels"][0]["wavelength_id"] == "A04_C01"
        assert zarr_group.attrs["omero"]["channels"][1]["wavelength_id"] == "A03_C02"

    # Check that the 2 expected OME-Zarr images exist after the task
    well_group = "/".join(search_first_ome_zarr_2d.split("/")[:-1])
    with zarr.open(well_group, mode="r") as zarr_group:
        assert list(zarr_group.group_keys()) == ["0", "0_fused"]


def test_stitching_overwrite(
    search_first_ome_zarr_2d,
    registration_resolution_level=1,
):
    image_list_updates = stitching_task(
        zarr_url=search_first_ome_zarr_2d,
        channel=StitchingChannelInputModel(wavelength_id="A04_C01"),
        registration_resolution_level=registration_resolution_level,
        overwrite_input=True,
    )
    assert image_list_updates is None
    well_group = "/".join(search_first_ome_zarr_2d.split("/")[:-1])
    with zarr.open(well_group, mode="r") as zarr_group:
        assert list(zarr_group.group_keys()) == ["0"]
