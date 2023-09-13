'''
A simpler setup version just to compile the speedup module.

It should be used as:

python setup_pydevd_cython build_ext --inplace

Note: the .c file and other generated files are regenerated from
the .pyx file by running "python build_tools/build.py"
'''

import os
import sys
from setuptools import setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

IS_PY36_OR_GREATER = sys.version_info > (3, 6)
TODO_PY311 = sys.version_info > (3, 11)


def process_args():
    extension_folder = None
    target_pydevd_name = None
    target_frame_eval = None
    force_cython = False

    for i, arg in enumerate(sys.argv[:]):
        if arg == '--build-lib':
            extension_folder = sys.argv[i + 1]
            # It shouldn't be removed from sys.argv (among with --build-temp) because they're passed further to setup()
        if arg.startswith('--target-pyd-name='):
            sys.argv.remove(arg)
            target_pydevd_name = arg[len('--target-pyd-name='):]
        if arg.startswith('--target-pyd-frame-eval='):
            sys.argv.remove(arg)
            target_frame_eval = arg[len('--target-pyd-frame-eval='):]
        if arg == '--force-cython':
            sys.argv.remove(arg)
            force_cython = True

    return extension_folder, target_pydevd_name, target_frame_eval, force_cython


def process_template_lines(template_lines):
    # Create 2 versions of the template, one for Python 3.8 and another for Python 3.9
    for version in ('38', '39'):
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'

        for line in template_lines:
            if version == '38':
                line = line.replace('get_bytecode_while_frame_eval(PyFrameObject * frame_obj, int exc)', 'get_bytecode_while_frame_eval_38(PyFrameObject * frame_obj, int exc)')
                line = line.replace('CALL_EvalFrameDefault', 'CALL_EvalFrameDefault_38(frame_obj, exc)')
            else:  # 3.9
                line = line.replace('get_bytecode_while_frame_eval(PyFrameObject * frame_obj, int exc)', 'get_bytecode_while_frame_eval_39(PyThreadState* tstate, PyFrameObject * frame_obj, int exc)')
                line = line.replace('CALL_EvalFrameDefault', 'CALL_EvalFrameDefault_39(tstate, frame_obj, exc)')

            yield line

        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield '### WARNING: GENERATED CODE, DO NOT EDIT!'
        yield ''
        yield ''


def process_template_file(contents):
    ret = []
    template_lines = []

    append_to = ret
    for line in contents.splitlines(keepends=False):
        if line.strip() == '### TEMPLATE_START':
            append_to = template_lines
        elif line.strip() == '### TEMPLATE_END':
            append_to = ret
            for line in process_template_lines(template_lines):
                ret.append(line)
        else:
            append_to.append(line)

    return '\n'.join(ret)


def build_extension(dir_name, extension_name, target_pydevd_name, force_cython, extended=False, has_pxd=False, template=False):
    pyx_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.pyx" % (extension_name,))

    if template:
        pyx_template_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.template.pyx" % (extension_name,))
        with open(pyx_template_file, 'r') as stream:
            contents = stream.read()

        contents = process_template_file(contents)

        with open(pyx_file, 'w') as stream:
            stream.write(contents)

    if target_pydevd_name != extension_name:
        # It MUST be there in this case!
        # (otherwise we'll have unresolved externals because the .c file had another name initially).
        import shutil

        # We must force cython in this case (but only in this case -- for the regular setup in the user machine, we
        # should always compile the .c file).
        force_cython = True

        new_pyx_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.pyx" % (target_pydevd_name,))
        new_c_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.c" % (target_pydevd_name,))
        shutil.copy(pyx_file, new_pyx_file)
        pyx_file = new_pyx_file
        if has_pxd:
            pxd_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.pxd" % (extension_name,))
            new_pxd_file = os.path.join(os.path.dirname(__file__), dir_name, "%s.pxd" % (target_pydevd_name,))
            shutil.copy(pxd_file, new_pxd_file)
        assert os.path.exists(pyx_file)

    try:
        c_files = [os.path.join(dir_name, "%s.c" % target_pydevd_name), ]
        if force_cython:
            for c_file in c_files:
                try:
                    os.remove(c_file)
                except:
                    pass
            from Cython.Build import cythonize  # @UnusedImport
            # Generate the .c files in cythonize (will not compile at this point).

            target = "%s/%s.pyx" % (dir_name, target_pydevd_name,)
            cythonize([target])

            # Workarounds needed in CPython 3.8 and 3.9 to access PyInterpreterState.eval_frame.
            for c_file in c_files:
                with open(c_file, 'r') as stream:
                    c_file_contents = stream.read()

                if '#include "internal/pycore_gc.h"' not in c_file_contents:
                    c_file_contents = c_file_contents.replace('#include "Python.h"', '''#include "Python.h"
#if PY_VERSION_HEX >= 0x03090000
#include "internal/pycore_gc.h"
#include "internal/pycore_interp.h"
#endif
''')

                if '#include "internal/pycore_pystate.h"' not in c_file_contents:
                    c_file_contents = c_file_contents.replace('#include "pystate.h"', '''#include "pystate.h"
#if PY_VERSION_HEX >= 0x03080000
#include "internal/pycore_pystate.h"
#endif
''')

                # We want the same output on Windows and Linux.
                c_file_contents = c_file_contents.replace('\r\n', '\n').replace('\r', '\n')
                c_file_contents = c_file_contents.replace(r'_pydevd_frame_eval\\release_mem.h', '_pydevd_frame_eval/release_mem.h')
                c_file_contents = c_file_contents.replace(r'_pydevd_frame_eval\\pydevd_frame_evaluator.pyx', '_pydevd_frame_eval/pydevd_frame_evaluator.pyx')
                c_file_contents = c_file_contents.replace(r'_pydevd_bundle\\pydevd_cython.pxd', '_pydevd_bundle/pydevd_cython.pxd')
                c_file_contents = c_file_contents.replace(r'_pydevd_bundle\\pydevd_cython.pyx', '_pydevd_bundle/pydevd_cython.pyx')

                with open(c_file, 'w') as stream:
                    stream.write(c_file_contents)

        # Always compile the .c (and not the .pyx) file (which we should keep up-to-date by running build_tools/build.py).
        from distutils.extension import Extension
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

        ext_modules = [
            Extension(
                "%s%s.%s" % (dir_name, "_ext" if extended else "", target_pydevd_name,),
                c_files,
                **kwargs
            )]

        # This is needed in CPython 3.8 to be able to include internal/pycore_pystate.h
        # (needed to set PyInterpreterState.eval_frame).
        for module in ext_modules:
            module.define_macros = [('Py_BUILD_CORE_MODULE', '1')]
        setup(
            name='Cythonize',
            ext_modules=ext_modules
        )
    finally:
        if target_pydevd_name != extension_name:
            try:
                os.remove(new_pyx_file)
            except:
                import traceback
                traceback.print_exc()
            try:
                os.remove(new_c_file)
            except:
                import traceback
                traceback.print_exc()
            if has_pxd:
                try:
                    os.remove(new_pxd_file)
                except:
                    import traceback
                    traceback.print_exc()


extension_folder, target_pydevd_name, target_frame_eval, force_cython = process_args()

extension_name = "pydevd_cython"
if target_pydevd_name is None:
    target_pydevd_name = extension_name
build_extension("_pydevd_bundle", extension_name, target_pydevd_name, force_cython, extension_folder, True)

if IS_PY36_OR_GREATER and not TODO_PY311:
    extension_name = "pydevd_frame_evaluator"
    if target_frame_eval is None:
        target_frame_eval = extension_name
    build_extension("_pydevd_frame_eval", extension_name, target_frame_eval, force_cython, extension_folder, True, template=True)

if extension_folder:
    os.chdir(extension_folder)
    for folder in [file for file in os.listdir(extension_folder) if
                   file != 'build' and os.path.isdir(os.path.join(extension_folder, file))]:
        file = os.path.join(folder, "__init__.py")
        if not os.path.exists(file):
            open(file, 'a').close()
