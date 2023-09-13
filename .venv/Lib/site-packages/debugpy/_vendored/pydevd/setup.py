r'''
Full setup, used to distribute the debugger backend to PyPi.

Note that this is mostly so that users can do:

pip install pydevd

in a machine for doing remote-debugging, as a local installation with the IDE should have
everything already distributed.

Reference on wheels:
https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/
http://lucumr.pocoo.org/2014/1/27/python-on-wheels/

Another (no wheels): https://jamie.curle.io/blog/my-first-experience-adding-package-pypi/

See:

build_tools\pydevd_release_process.txt

for release process.
'''

from setuptools import setup
from setuptools.dist import Distribution
from distutils.extension import Extension
import os


class BinaryDistribution(Distribution):

    def is_pure(self):
        return False


data_files = []


def accept_file(f):
    f = f.lower()
    for ext in '.py .dll .so .dylib .txt .cpp .h .bat .c .sh .md .txt'.split():
        if f.endswith(ext):
            return True

    return f in ['readme', 'makefile']


data_files.append(('pydevd_attach_to_process', [os.path.join('pydevd_attach_to_process', f) for f in os.listdir('pydevd_attach_to_process') if accept_file(f)]))
for root, dirs, files in os.walk("pydevd_attach_to_process"):
    for d in dirs:
        data_files.append((os.path.join(root, d), [os.path.join(root, d, f) for f in os.listdir(os.path.join(root, d)) if accept_file(f)]))

import pydevd
version = pydevd.__version__

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), 'r', encoding='utf-8') as stream:
    long_description = stream.read()

args = dict(
    name='pydevd',
    version=version,
    description='PyDev.Debugger (used in PyDev, PyCharm and VSCode Python)',
    long_description_content_type='text/markdown',
    long_description=long_description,
    author='Fabio Zadrozny and others',
    url='https://github.com/fabioz/PyDev.Debugger/',
    license='EPL, Apache 2.0',
    packages=[
        '_pydev_bundle',
        '_pydev_bundle.fsnotify',
        '_pydev_runfiles',
        '_pydevd_bundle',
        '_pydevd_bundle._debug_adapter',
        '_pydevd_bundle.pydevd_concurrency_analyser',
        '_pydevd_frame_eval',
        '_pydevd_frame_eval.vendored',
        '_pydevd_frame_eval.vendored.bytecode',
        'pydev_ipython',

        # 'pydev_sitecustomize', -- Not actually a package (not added)

        'pydevd_attach_to_process',

        'pydevd_plugins',
        'pydevd_plugins.extensions',
        'pydevd_plugins.extensions.types',
    ],
    py_modules=[
        # 'interpreterInfo', -- Not needed for debugger
        # 'pycompletionserver', -- Not needed for debugger
        'pydev_app_engine_debug_startup',
        # 'pydev_coverage', -- Not needed for debugger
        # 'pydev_pysrc', -- Not needed for debugger
        'pydev_run_in_console',
        'pydevconsole',
        'pydevd_file_utils',
        'pydevd',
        'pydevd_tracing',
        # 'runfiles', -- Not needed for debugger
        'setup_pydevd_cython',  # Distributed to clients. See: https://github.com/fabioz/PyDev.Debugger/issues/102
        # 'setup', -- Should not be included as a module
    ],
    classifiers=[
        'Development Status :: 6 - Mature',
        'Environment :: Console',
        'Intended Audience :: Developers',

        'License :: OSI Approved :: Eclipse Public License 1.0 (EPL-1.0)',
        'License :: OSI Approved :: Apache Software License',

        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Debuggers',
    ],
    entry_points={
        'console_scripts':[
            'pydevd = pydevd:main',
        ],
    },
    data_files=data_files,
    keywords=['pydev', 'pydevd', 'pydev.debugger'],
    include_package_data=True,
    zip_safe=False,
)

import sys
try:
    extra_compile_args = []
    extra_link_args = []

    if 'linux' in sys.platform:
        # Enabling -flto brings executable from 4MB to 0.56MB and -Os to 0.41MB
        # Profiling shows an execution around 3-5% slower with -Os vs -O3,
        # so, kept only -flto.
        extra_compile_args = ["-flto", "-O3"]
        extra_link_args = extra_compile_args[:]

        # Note: also experimented with profile-guided optimization. The executable
        # size became a bit smaller (from 0.56MB to 0.5MB) but this would add an
        # extra step to run the debugger to obtain the optimizations
        # so, skipped it for now (note: the actual benchmarks time was in the
        # margin of a 0-1% improvement, which is probably not worth it for
        # speed increments).
        # extra_compile_args = ["-flto", "-fprofile-generate"]
        # ... Run benchmarks ...
        # extra_compile_args = ["-flto", "-fprofile-use", "-fprofile-correction"]
    elif 'win32' in sys.platform:
        pass
        # uncomment to generate pdbs for visual studio.
        # extra_compile_args=["-Zi", "/Od"]
        # extra_link_args=["-debug"]

    kwargs = {}
    if extra_link_args:
        kwargs['extra_link_args'] = extra_link_args
    if extra_compile_args:
        kwargs['extra_compile_args'] = extra_compile_args

    args_with_binaries = args.copy()
    args_with_binaries.update(dict(
        distclass=BinaryDistribution,
        ext_modules=[
            # In this setup, don't even try to compile with cython, just go with the .c file which should've
            # been properly generated from a tested version.
            Extension(
                '_pydevd_bundle.pydevd_cython',
                ["_pydevd_bundle/pydevd_cython.c", ],
                define_macros=[('Py_BUILD_CORE_MODULE', '1')],
                **kwargs
            )
        ]
    ))
    setup(**args_with_binaries)
except:
    # Compile failed: just setup without compiling cython deps.
    setup(**args)
    sys.stdout.write('Plain-python version of pydevd installed (cython speedups not available).\n')
