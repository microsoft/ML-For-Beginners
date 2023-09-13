'''
Helper to build pydevd.

It should:
    * recreate our generated files
    * compile cython deps (properly setting up the environment first).

Note that it's used in the CI to build the cython deps based on the PYDEVD_USE_CYTHON environment variable.
'''
from __future__ import print_function

import os
import subprocess
import sys

from generate_code import remove_if_exists, root_dir, is_python_64bit, generate_dont_trace_files, generate_cython_module


def validate_pair(ob):
    try:
        if not (len(ob) == 2):
            print("Unexpected result:", ob, file=sys.stderr)
            raise ValueError
    except:
        return False
    return True


def consume(it):
    try:
        while True:
            next(it)
    except StopIteration:
        pass


def get_environment_from_batch_command(env_cmd, initial=None):
    """
    Take a command (either a single command or list of arguments)
    and return the environment created after running that command.
    Note that if the command must be a batch file or .cmd file, or the
    changes to the environment will not be captured.

    If initial is supplied, it is used as the initial environment passed
    to the child process.
    """
    if not isinstance(env_cmd, (list, tuple)):
        env_cmd = [env_cmd]
    if not os.path.exists(env_cmd[0]):
        raise RuntimeError('Error: %s does not exist' % (env_cmd[0],))

    # construct the command that will alter the environment
    env_cmd = subprocess.list2cmdline(env_cmd)
    # create a tag so we can tell in the output when the proc is done
    tag = 'Done running command'
    # construct a cmd.exe command to do accomplish this
    cmd = 'cmd.exe /s /c "{env_cmd} && echo "{tag}" && set"'.format(**vars())
    # launch the process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=initial)
    # parse the output sent to stdout
    lines = proc.stdout
    # consume whatever output occurs until the tag is reached
    for line in lines:
        line = line.decode('utf-8')
        if 'The specified configuration type is missing.' in line:
            raise AssertionError('Error executing %s. View http://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/ for details.' % (env_cmd))
        if tag in line:
            break
    if sys.version_info[0] > 2:
        # define a way to handle each KEY=VALUE line
        handle_line = lambda l: l.decode('utf-8').rstrip().split('=', 1)
    else:
        # define a way to handle each KEY=VALUE line
        handle_line = lambda l: l.rstrip().split('=', 1)
    # parse key/values into pairs
    pairs = map(handle_line, lines)
    # make sure the pairs are valid
    valid_pairs = filter(validate_pair, pairs)
    # construct a dictionary of the pairs
    result = dict(valid_pairs)
    # let the process finish
    proc.communicate()
    return result


def remove_binaries(suffixes):
    for f in os.listdir(os.path.join(root_dir, '_pydevd_bundle')):
        for suffix in suffixes:
            if f.endswith(suffix):
                remove_if_exists(os.path.join(root_dir, '_pydevd_bundle', f))


def build():
    if '--no-remove-binaries' not in sys.argv:
        remove_binaries(['.pyd', '.so'])

    os.chdir(root_dir)

    env = None
    if sys.platform == 'win32':
        # "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars64.bat"
        # set MSSdk=1
        # set DISTUTILS_USE_SDK=1
        # set VS100COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 9.0\Common7\Tools

        if 'GITHUB_ACTION' not in os.environ:
            env = os.environ.copy()
            if sys.version_info[:2] in ((3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11)):
                FORCE_PYDEVD_VC_VARS = os.environ.get('FORCE_PYDEVD_VC_VARS')
                if FORCE_PYDEVD_VC_VARS:
                    env.update(get_environment_from_batch_command([FORCE_PYDEVD_VC_VARS], initial=os.environ.copy()))
                else:
                    import setuptools  # We have to import it first for the compiler to be found
                    from distutils import msvc9compiler

                    vcvarsall = msvc9compiler.find_vcvarsall(14.0)
                    if vcvarsall is None or not os.path.exists(vcvarsall):
                        msvc_version = msvc9compiler.get_build_version()
                        print('msvc_version', msvc_version)
                        vcvarsall = msvc9compiler.find_vcvarsall(msvc_version)

                    if vcvarsall is None or not os.path.exists(vcvarsall):
                        raise RuntimeError('Error finding vcvarsall.')

                    if is_python_64bit():
                        env.update(get_environment_from_batch_command(
                            [vcvarsall, 'amd64'],
                            initial=os.environ.copy()))
                    else:
                        env.update(get_environment_from_batch_command(
                            [vcvarsall, 'x86'],
                            initial=os.environ.copy()))

            else:
                raise AssertionError('Unable to setup environment for Python: %s' % (sys.version,))

            env['MSSdk'] = '1'
            env['DISTUTILS_USE_SDK'] = '1'

    additional_args = []
    for arg in sys.argv:
        if arg.startswith('--target-pyd-name='):
            additional_args.append(arg)
        if arg.startswith('--target-pyd-frame-eval='):
            additional_args.append(arg)
            break
    else:
        additional_args.append('--force-cython')  # Build always forces cython!

    args = [
        sys.executable, os.path.join(os.path.dirname(__file__), '..', 'setup_pydevd_cython.py'), 'build_ext', '--inplace',
    ] + additional_args
    print('Calling args: %s' % (args,))
    subprocess.check_call(args, env=env,)


if __name__ == '__main__':
    use_cython = os.getenv('PYDEVD_USE_CYTHON', '').lower()
    # Note: don't import pydevd during build (so, accept just yes/no in this case).
    if use_cython == 'yes':
        print("Building")
        build()
    elif use_cython == 'no':
        print("Removing binaries")
        remove_binaries(['.pyd', '.so'])
    elif not use_cython:
        # Regular process
        if '--no-regenerate-files' not in sys.argv:
            print("Generating dont trace files")
            generate_dont_trace_files()
            print("Generating cython modules")
            generate_cython_module()
        print("Building")
        build()
    else:
        raise RuntimeError('Unexpected value for PYDEVD_USE_CYTHON: %s (accepted: yes, no)' % (use_cython,))

