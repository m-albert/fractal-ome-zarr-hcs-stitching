import shutil, os
from pathlib import Path

import pytest
from devtools import debug

import anndata as ad

from fractal_ome_zarr_hcs_stitching.stitching_task import stitching_task


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """

    # the test data from the fractal task template has only two z planes,
    # which makes it suboptimal for registering in 3D. It's however ok
    # for testing the correctness of the stitching task in 3D.
    # Ideally, we would stream different datasets for 2D and 3D tests here.
    source_dir = (Path(__file__).parent / "data/ngff_example/my_image").as_posix()
    # source_dir = (Path(__file__).parent / "data/240225shape_mip.zarr/A/01/0").as_posix()
    # source_dir = (Path(__file__).parent / "data/240225shape.zarr/A/01/0").as_posix()

    dest_dir = (tmp_path / "my_image").as_posix()
    # save to the same folder as the source to inspect the results
    # dest_dir = (Path(__file__).parent / "data/ngff_example/my_image_test_copy").as_posix()
    # if os.path.exists(dest_dir): shutil.rmtree(dest_dir)
    
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)

    # Create some overlap between the FOVs
    fov_roi_table_path = os.path.join(dest_dir, 'tables/FOV_ROI_table')
    fov_roi_table = ad.read_zarr(fov_roi_table_path)
    fov_roi_table.X[1][6] -= 7
    fov_roi_table.write_zarr(fov_roi_table_path)
    
    return dest_dir


def test_stitching_task(test_data_dir):
    stitching_task(
        zarr_url=test_data_dir,
        registration_channel_label="DAPI",
        registration_binning_xy=1,
        registration_binning_z=1,
        )
