"""Contains the list of tasks available to fractal."""

from fractal_tasks_core.dev.task_models import ParallelTask

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
