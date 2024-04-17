import json
import subprocess
from shlex import split as shlex_split
from subprocess import PIPE

import pytest
from devtools import debug

from . import MANIFEST
from . import PACKAGE_DIR


def validate_command(cmd: str):
    """
    Run a command and make assertions on stdout, stderr, retcode.
    """
    debug(cmd)
    result = subprocess.run(  # nosec
        shlex_split(cmd),
        stdout=PIPE,
        stderr=PIPE,
    )
    # The command must always fail, since tmp_file_args includes invalid
    # arguments
    assert result.returncode == 1
    stderr = result.stderr.decode()
    debug(stderr)

    # Valid stderr includes pydantic.error_wrappers.ValidationError (type
    # match between model and function, but tmp_file_args has wrong arguments)
    assert "pydantic.error_wrappers.ValidationError" in stderr

    # Valid stderr must include a mention of "unexpected keyword arguments",
    # because we are including some invalid arguments
    assert "unexpected keyword arguments" in stderr

    # Invalid stderr includes ValueError
    assert "ValueError" not in stderr


@pytest.mark.parametrize("task", MANIFEST["task_list"])
def test_task_interface(task, tmp_path):
    """
    Test that running tasks from the command line with invalid arguments leads
    to the expected behavior.
    """
    tmp_file_args = str(tmp_path / "args.json")
    tmp_file_metadiff = str(tmp_path / "metadiff.json")
    with open(tmp_file_args, "w") as fout:
        args = dict(wrong_arg_1=123, wrong_arg_2=[1, 2, 3])
        json.dump(args, fout, indent=4)

    for key in ["executable_non_parallel", "executable_parallel"]:
        executable = task.get(key)
        if executable is None:
            continue
        task_path = (PACKAGE_DIR / executable).as_posix()
        cmd = (
            f"python {task_path} "
            f"--args-json {tmp_file_args} "
            f"--out-json {tmp_file_metadiff}"
        )
        validate_command(cmd)
