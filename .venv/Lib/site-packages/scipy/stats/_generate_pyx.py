import pathlib
import subprocess
import sys
import os
import argparse


def make_boost(outdir, distutils_build=False):
    # Call code generator inside _boost directory
    code_gen = pathlib.Path(__file__).parent / '_boost/include/code_gen.py'
    if distutils_build:
        subprocess.run([sys.executable, str(code_gen), '-o', outdir,
                        '--distutils-build', 'True'], check=True)
    else:
        subprocess.run([sys.executable, str(code_gen), '-o', outdir],
                       check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", type=str,
                        help="Path to the output directory")
    args = parser.parse_args()

    if not args.outdir:
        # We're dealing with a distutils build here, write in-place:
        outdir_abs = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
        outdir_abs_boost = outdir_abs / '_boost' / 'src'
        if not os.path.exists(outdir_abs_boost):
            os.makedirs(outdir_abs_boost)
        make_boost(outdir_abs_boost, distutils_build=True)
    else:
        # Meson build
        srcdir_abs = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
        outdir_abs = pathlib.Path(os.getcwd()) / args.outdir
        make_boost(outdir_abs)
