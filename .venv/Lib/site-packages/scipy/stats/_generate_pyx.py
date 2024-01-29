import pathlib
import subprocess
import sys
import os
import argparse


def make_boost(outdir):
    # Call code generator inside _boost directory
    code_gen = pathlib.Path(__file__).parent / '_boost/include/code_gen.py'
    subprocess.run([sys.executable, str(code_gen), '-o', outdir],
                   check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    args = parser.parse_args()

    if not args.outdir:
        raise ValueError("A path to the output directory is required")
    else:
        # Meson build
        srcdir_abs = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
        outdir_abs = pathlib.Path(os.getcwd()) / args.outdir
        make_boost(outdir_abs)
