
from __future__ import unicode_literals

import os
import subprocess
import sys

miniconda64_envs = os.getenv('MINICONDA64_ENVS')
python_installations = [
    r'%s/py34_64/bin/python' % miniconda64_envs,
    r'%s/py35_64/bin/python' % miniconda64_envs,
    r'%s/py36_64/bin/python' % miniconda64_envs,
    r'%s/py37_64/bin/python' % miniconda64_envs,
    ]
root_dir = os.path.dirname(os.path.dirname(__file__))


def list_binaries():
    for f in os.listdir(os.path.join(root_dir, '_pydevd_bundle')):
        if f.endswith('.so'):
            yield f


def extract_version(python_install):
    return python_install.split('/')[-3][2:]


def main():
    from generate_code import generate_dont_trace_files
    from generate_code import generate_cython_module

    # First, make sure that our code is up to date.
    generate_dont_trace_files()
    generate_cython_module()

    for python_install in python_installations:
        assert os.path.exists(python_install)

    from build import remove_binaries
    remove_binaries(['.so'])

    for f in list_binaries():
        raise AssertionError('Binary not removed: %s' % (f,))

    for i, python_install in enumerate(python_installations):
        new_name = 'pydevd_cython_%s_%s' % (sys.platform, extract_version(python_install))
        args = [
            python_install, os.path.join(root_dir, 'build_tools', 'build.py'), '--no-remove-binaries', '--target-pyd-name=%s' % new_name, '--force-cython']
        if i != 0:
            args.append('--no-regenerate-files')
        version_number = extract_version(python_install)
        if version_number.startswith('36') or version_number.startswith('37'):
            name_frame_eval = 'pydevd_frame_evaluator_%s_%s' % (sys.platform, extract_version(python_install))
            args.append('--target-pyd-frame-eval=%s' % name_frame_eval)
        print('Calling: %s' % (' '.join(args)))
        subprocess.check_call(args)


if __name__ == '__main__':
    main()
