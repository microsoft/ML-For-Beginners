#!/usr/bin/env python
"""
display environment information that is frequently
used to troubleshoot installations of Jupyter or IPython
"""

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union


def subs(cmd: Union[List[str], str]) -> Optional[str]:
    """
    get data from commands that we need to run outside of python
    """
    try:
        stdout = subprocess.check_output(cmd)  # noqa
        return stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def get_data() -> Dict[str, Any]:
    """
    returns a dict of various user environment data
    """
    env: Dict[str, Any] = {}
    env["path"] = os.environ.get("PATH")
    env["sys_path"] = sys.path
    env["sys_exe"] = sys.executable
    env["sys_version"] = sys.version
    env["platform"] = platform.platform()
    # FIXME: which on Windows?
    if sys.platform == "win32":
        env["where"] = subs(["where", "jupyter"])
        env["which"] = None
    else:
        env["which"] = subs(["which", "-a", "jupyter"])
        env["where"] = None
    env["pip"] = subs([sys.executable, "-m", "pip", "list"])
    env["conda"] = subs(["conda", "list"])
    env["conda-env"] = subs(["conda", "env", "export"])
    return env


def main() -> None:  # noqa
    """
    print out useful info
    """
    # pylint: disable=superfluous-parens
    # args = get_args()
    if "_ARGCOMPLETE" in os.environ:
        # No arguments to complete, the script can be slow to run to completion,
        # so in case someone tries to complete jupyter troubleshoot just exit early
        return

    environment_data = get_data()

    print("$PATH:")
    for directory in environment_data["path"].split(os.pathsep):
        print(f"\t{directory}")

    print("\nsys.path:")
    for directory in environment_data["sys_path"]:
        print(f"\t{directory}")

    print("\nsys.executable:")
    print(f'\t{environment_data["sys_exe"]}')

    print("\nsys.version:")
    if "\n" in environment_data["sys_version"]:
        for data in environment_data["sys_version"].split("\n"):
            print(f"\t{data}")
    else:
        print(f'\t{environment_data["sys_version"]}')

    print("\nplatform.platform():")
    print(f'\t{environment_data["platform"]}')

    if environment_data["which"]:
        print("\nwhich -a jupyter:")
        for line in environment_data["which"].split("\n"):
            print(f"\t{line}")

    if environment_data["where"]:
        print("\nwhere jupyter:")
        for line in environment_data["where"].split("\n"):
            print(f"\t{line}")

    if environment_data["pip"]:
        print("\npip list:")
        for package in environment_data["pip"].split("\n"):
            print(f"\t{package}")

    if environment_data["conda"]:
        print("\nconda list:")
        for package in environment_data["conda"].split("\n"):
            print(f"\t{package}")

    if environment_data["conda-env"]:
        print("\nconda env:")
        for package in environment_data["conda-env"].split("\n"):
            print(f"\t{package}")


if __name__ == "__main__":
    main()
