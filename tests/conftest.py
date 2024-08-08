import os
import shutil
from pathlib import Path

import pooch
import pytest


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_tiled_ome_zarr(testdata_path: Path) -> str:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.12795605"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / DOI_slug

    registry = {
        "fractal_output.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    file_name = "fractal_output"
    # 1) Download the fractal output folder from Zenodo
    file_paths = POOCH.fetch(
        f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
    )
    zarr_full_path = file_paths[0].split(file_name)[0] + file_name
    print(zarr_full_path)

    # 2) Copy the downloaded Zarr into tests/data
    if os.path.isdir(str(rootfolder)):
        shutil.rmtree(str(rootfolder))
    shutil.copytree(Path(zarr_full_path) / file_name, rootfolder)
    return rootfolder


@pytest.fixture(scope="session")
def zenodo_search_first_ome_zarr(testdata_path: Path) -> str:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.12795788"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    rootfolder = testdata_path / DOI_slug

    registry = {
        "fractal_output.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )

    # 1) Download the fractal output folder from Zenodo
    file_name = "fractal_output"
    file_paths = POOCH.fetch(
        f"{file_name}.zip", processor=pooch.Unzip(extract_dir=file_name)
    )
    zarr_full_path = file_paths[0].split(file_name)[0] + file_name

    # 2) Copy the downloaded Zarr into tests/data
    if os.path.isdir(str(rootfolder)):
        shutil.rmtree(str(rootfolder))
    shutil.copytree(Path(zarr_full_path) / file_name, rootfolder)
    return rootfolder


@pytest.fixture(scope="function")
def tiled_ome_zarr_2d(tmpdir, zenodo_tiled_ome_zarr) -> str:
    plate_name = "231129NAR_mip.zarr"
    zarr_plate_path = f"{zenodo_tiled_ome_zarr}/{plate_name}"
    target_dir = tmpdir / plate_name
    if os.path.isdir(str(target_dir)):
        shutil.rmtree(str(target_dir))
    shutil.copytree(zarr_plate_path, target_dir)
    return f"{target_dir}/B/02/0"


@pytest.fixture(scope="function")
def tiled_ome_zarr_3d(tmpdir, zenodo_tiled_ome_zarr) -> str:
    plate_name = "231129NAR.zarr"
    zarr_plate_path = f"{zenodo_tiled_ome_zarr}/{plate_name}"
    target_dir = tmpdir / plate_name
    if os.path.isdir(str(target_dir)):
        shutil.rmtree(str(target_dir))
    shutil.copytree(zarr_plate_path, target_dir)
    return f"{target_dir}/B/02/0"


@pytest.fixture(scope="function")
def search_first_ome_zarr_2d(tmpdir, zenodo_search_first_ome_zarr) -> str:
    plate_name = "dcflexr1_mip.zarr"
    zarr_plate_path = f"{zenodo_search_first_ome_zarr}/{plate_name}"
    target_dir = tmpdir / plate_name
    if os.path.isdir(str(target_dir)):
        shutil.rmtree(str(target_dir))
    shutil.copytree(Path(zarr_plate_path), target_dir)
    return f"{target_dir}/C/05/0"


@pytest.fixture(scope="function")
def search_first_ome_zarr_3d(tmpdir, zenodo_search_first_ome_zarr) -> str:
    plate_name = "dcflexr1.zarr"
    zarr_plate_path = f"{zenodo_search_first_ome_zarr}/{plate_name}"
    target_dir = tmpdir / plate_name
    if os.path.isdir(str(target_dir)):
        shutil.rmtree(str(target_dir))
    shutil.copytree(Path(zarr_plate_path), target_dir)
    return f"{target_dir}/C/05/0"
