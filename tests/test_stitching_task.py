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
    source_dir = (Path(__file__).parent / "data/ngff_example/my_image").as_posix()
    # source_dir = Path("/Users/malbert/projects/nrepina/20240225_shape_remapped_slide4_small/fractal_output/20240225_shape_small_output/240225shape_mip.zarr/A/01/0").as_posix()
    dest_dir = (tmp_path / "my_image").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)

    # Create some overlap between the FOVs
    fov_roi_table_path = os.path.join(dest_dir, 'tables/FOV_ROI_table')
    fov_roi_table = ad.read_zarr(fov_roi_table_path)
    fov_roi_table.X[1][6] -= 100
    fov_roi_table.write_zarr(fov_roi_table_path)
    
    return dest_dir


def test_stitching_task(test_data_dir):
    stitching_task(zarr_url=test_data_dir)
