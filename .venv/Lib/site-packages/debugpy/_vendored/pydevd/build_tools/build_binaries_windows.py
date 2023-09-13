r'''
Creating the needed environments for creating the pre-compiled distribution on Windows:

See:

build_tools\pydevd_release_process.txt

for building binaries/release process.
'''

from __future__ import unicode_literals
import os
import subprocess
import sys

miniconda_envs = os.getenv('MINICONDA_ENVS', r'C:\bin\Miniconda3\envs')

python_installations = [
    r'%s\py36_64\python.exe' % miniconda_envs,
    r'%s\py37_64\python.exe' % miniconda_envs,
    r'%s\py38_64\python.exe' % miniconda_envs,
    r'%s\py39_64\python.exe' % miniconda_envs,
    r'%s\py310_64\python.exe' % miniconda_envs,
    r'%s\py311_64\python.exe' % miniconda_envs,
]

root_dir = os.path.dirname(os.path.dirname(__file__))


def list_binaries():
    for f in os.listdir(os.path.join(root_dir, '_pydevd_bundle')):
        if f.endswith('.pyd'):
            yield f


def extract_version(python_install):
    return python_install.split('\\')[-2][2:]


def main():
    from generate_code import generate_dont_trace_files
    from generate_code import generate_cython_module

    # First, make sure that our code is up to date.
    generate_dont_trace_files()
    generate_cython_module()

    for python_install in python_installations:
        assert os.path.exists(python_install), '%s does not exist.' % (python_install,)

    from build import remove_binaries
    remove_binaries(['.pyd'])

    for f in list_binaries():
        raise AssertionError('Binary not removed: %s' % (f,))

    for i, python_install in enumerate(python_installations):
        print()
        print('*' * 80)
        print('*' * 80)
        print()
        new_name = 'pydevd_cython_%s_%s' % (sys.platform, extract_version(python_install))
        args = [
            python_install, os.path.join(root_dir, 'build_tools', 'build.py'), '--no-remove-binaries', '--target-pyd-name=%s' % new_name, '--force-cython']
        if i != 0:
            args.append('--no-regenerate-files')
        name_frame_eval = 'pydevd_frame_evaluator_%s_%s' % (sys.platform, extract_version(python_install))
        args.append('--target-pyd-frame-eval=%s' % name_frame_eval)
        print('Calling: %s' % (' '.join(args)))

        env = os.environ.copy()
        python_exe_dir = os.path.dirname(python_install)
        env['PATH'] = env['PATH'] + ';' + os.path.join(python_exe_dir, 'DLLs') + ';' + os.path.join(python_exe_dir, 'Library', 'bin')
        subprocess.check_call(args, env=env)


if __name__ == '__main__':
    main()

# To run do:
# See:
#
# build_tools\pydevd_release_process.txt
#
# for building binaries/release process.
