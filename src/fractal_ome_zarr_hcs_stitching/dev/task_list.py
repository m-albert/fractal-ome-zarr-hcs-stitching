"""Contains the list of tasks available to fractal."""

from fractal_task_tools.task_models import ParallelTask

AUTHORS = "Marvin Albert, Joel Luethi, Nicole Repina"
DOCS_LINK = "https://github.com/m-albert/fractal-ome-zarr-hcs-stitching"
INPUT_MODELS = [
    ["fractal_ome_zarr_hcs_stitching", "utils.py", "StitchingChannelInputModel"],
]
TASK_LIST = [
    ParallelTask(
        name="Stitching Task",
        input_types={"stitched": False},
        output_types={"stitched": True},
        executable="stitching_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        category="Registration",
        tags=["multiview-stitcher", "Fusion", "Registration", "Stitching", "2D", "3D"],
    ),
]
