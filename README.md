# fractal-ome-zarr-hcs-stitching

Fractal task(s) for registering and fusing OME-Zarr HCS

## Development instructions

This instructions are only relevant *after* you completed both the `copier
copy` command and the git/GitLab/GitHub initialization phase - see
[README](https://github.com/fractal-analytics-platform/fractal-tasks-template#readme)
for details.

1. It is recommended to work from an isolated Python virtual environment:
```console
# Create the virtual environment in the folder venv
python -m venv venv
# Activate the Python virtual environment
source venv/bin/activate
# Deactivate the virtual environment, when you don't need it any more
deactivate
```
2. You can install your package locally as in:
```console
# Install only fractal_ome_zarr_hcs_stitching:
python -m pip install -e .
# Install both fractal_ome_zarr_hcs_stitching and development dependencies (e.g. pytest):
python -m pip install -e ".[dev]"
```

3. Enjoy developing the package.

4. The template already includes a sample task ("Thresholding Task"). Whenever
you change its input parameters or docstring, re-run
```console
python src/fractal_ome_zarr_hcs_stitching/dev/create_manifest.py
git add src/fractal_ome_zarr_hcs_stitching/__FRACTAL_MANIFEST__.json
git commit -m'Update `__FRACTAL_MANIFEST__.json`'
git push origin main
```

5. If you add a new task, you should also add a new item to the `TASK_LIST`
list, in `src/fractal_ome_zarr_hcs_stitching/dev/task_list.py`. Here is an example:
```python
from fractal_tasks_core.dev.task_models import NonParallelTask
from fractal_tasks_core.dev.task_models import ParallelTask
from fractal_tasks_core.dev.task_models import CompoundTask


TASK_LIST = [
    NonParallelTask(
        name="My non-parallel task",
        executable="my_non_parallel_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    ParallelTask(
        name="My parallel task",
        executable="my_parallel_task.py",
        meta={"cpus_per_task": 1, "mem": 4000},
    ),
    CompoundTask(
        name="My compound task",
        executable_init="my_task_init.py",
        executable="my_actual_task.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        meta={"cpus_per_task": 2, "mem": 12000},
    ),
]
```
Notes:

* After adding a task, you should also update the manifest (see point 4 above).
* The minimal example above also includes the `meta` and/or `meta_init` task properties; these are optional, and you can remove them if not needed.
* More details on Fractal tasks will be soon available at https://fractal-analytics-platform.github.io.

6. Run the test suite (with somewhat verbose logging) through
```console
python -m pytest --log-cli-level info -s
```
7. Build the package through
```console
python -m build
```
This command will create the release distribution files in the `dist` folder.
The wheel one (ending with `.whl`) is the one you can use to collect your tasks
within Fractal.
