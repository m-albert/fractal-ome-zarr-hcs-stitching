from fractal_tasks_core.dev.task_models import ParallelTask

TASK_LIST = [
    ParallelTask(
        name="Thresholding Task",
        executable="stitching_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
]