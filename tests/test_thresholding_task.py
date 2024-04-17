import shutil
from pathlib import Path

import pytest
from devtools import debug

from fractal_ome_zarr_hcs_stitching.stitching_task import stitching_task


@pytest.fixture(scope="function")
def test_data_dir(tmp_path: Path) -> str:
    """
    Copy a test-data folder into a temporary folder.
    """
    source_dir = (Path(__file__).parent / "data/ngff_example/my_image").as_posix()
    dest_dir = (tmp_path / "my_image").as_posix()
    debug(source_dir, dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def test_stitching_task(test_data_dir):
    stitching_task(zarr_url=test_data_dir)
