# SPDX-License-Identifier: Apache-2.0
import sys
from textwrap import dedent


def _help():
    print(
        dedent(
            """
        python -m skl2onnx [command]

    command is:

    setup   generate rst documentation for every ONNX operator
            before building the package"""
        )
    )


def _setup():
    from skl2onnx.algebra.onnx_ops import dynamic_class_creation

    dynamic_class_creation(True)


def main(argv):
    if len(argv) <= 1 or "--help" in argv:
        _help()
        return

    if "setup" in argv:
        print("generate rst documentation for every ONNX operator")
        _setup()
        return

    _help()


if __name__ == "__main__":
    main(sys.argv)
