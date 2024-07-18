import json
from pathlib import Path

import pytest
from fractal_tasks_core.dev.lib_args_schemas import (
    create_schema_for_single_task,
)
from fractal_tasks_core.dev.lib_signature_constraints import (
    _extract_function,
    _validate_function_signature,
)
from jsonschema.validators import (
    Draft7Validator,
    Draft201909Validator,
    Draft202012Validator,
)

from . import TASK_LIST


def test_task_functions_have_valid_signatures():
    """
    Test that task functions have valid signatures.
    """
    for task in TASK_LIST:
        for key in ["executable_non_parallel", "executable_parallel"]:
            executable = task.get(key)
            if executable is None:
                continue
            function_name = Path(executable).with_suffix("").name
            task_function = _extract_function(
                executable, function_name, package_name="fractal_ome_zarr_hcs_stitching"
            )
            _validate_function_signature(task_function)


def test_args_schemas_are_up_to_date():
    """
    Test that args_schema attributes in the manifest are up-to-date
    """
    for task in TASK_LIST:
        for kind in ["_non_parallel", "_parallel"]:
            executable = task.get(f"executable_{kind}")
            if executable is None:
                continue
            print(f"Now handling {executable}")
            old_schema = task[f"args_schema_{kind}"]
            new_schema = create_schema_for_single_task(
                executable, package="fractal_ome_zarr_hcs_stitching"
            )
            # The following step is required because some arguments may have a
            # default which has a non-JSON type (e.g. a tuple), which we need
            # to convert to JSON type (i.e. an array) before comparison.
            new_schema = json.loads(json.dumps(new_schema))
            assert new_schema == old_schema


@pytest.mark.parametrize(
    "jsonschema_validator",
    [Draft7Validator, Draft201909Validator, Draft202012Validator],
)
def test_args_schema_comply_with_jsonschema_specs(jsonschema_validator):
    """
    This test is actually useful, see
    https://github.com/fractal-analytics-platform/fractal-tasks-core/issues/564.
    """
    for task in TASK_LIST:
        for kind in ["_non_parallel", "_parallel"]:
            executable = task.get(f"executable_{kind}")
            if executable is None:
                continue
            print(f"Now handling {executable}")
            schema = task[f"args_schema_{kind}"]
            my_validator = jsonschema_validator(schema=schema)
            my_validator.check_schema(my_validator.schema)
            print(
                f"Schema for task {task['executable']} is valid for "
                f"{jsonschema_validator}."
            )
