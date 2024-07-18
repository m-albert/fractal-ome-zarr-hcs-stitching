import json
from pathlib import Path

import fractal_ome_zarr_hcs_stitching

PACKAGE = "fractal_ome_zarr_hcs_stitching"
PACKAGE_DIR = Path(fractal_ome_zarr_hcs_stitching.__file__).parent
MANIFEST_FILE = PACKAGE_DIR / "__FRACTAL_MANIFEST__.json"
with MANIFEST_FILE.open("r") as f:
    MANIFEST = json.load(f)
    TASK_LIST = MANIFEST["task_list"]
