# Defines which version of the trace_dispatch we'll use.
# Should give warning only here if cython is not available but supported.

import os
from _pydevd_bundle.pydevd_constants import USE_CYTHON_FLAG, ENV_TRUE_LOWER_VALUES, \
    ENV_FALSE_LOWER_VALUES
from _pydev_bundle import pydev_log

dirname = os.path.dirname(os.path.dirname(__file__))
USING_CYTHON = False


def delete_old_compiled_extensions():
    import _pydevd_bundle
    cython_extensions_dir = os.path.dirname(os.path.dirname(_pydevd_bundle.__file__))
    _pydevd_bundle_ext_dir = os.path.dirname(_pydevd_bundle.__file__)
    _pydevd_frame_eval_ext_dir = os.path.join(cython_extensions_dir, '_pydevd_frame_eval_ext')
    try:
        import shutil
        for file in os.listdir(_pydevd_bundle_ext_dir):
            if file.startswith("pydevd") and file.endswith(".so"):
                os.remove(os.path.join(_pydevd_bundle_ext_dir, file))
        for file in os.listdir(_pydevd_frame_eval_ext_dir):
            if file.startswith("pydevd") and file.endswith(".so"):
                os.remove(os.path.join(_pydevd_frame_eval_ext_dir, file))
        build_dir = os.path.join(cython_extensions_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(os.path.join(cython_extensions_dir, "build"))
    except OSError:
        pydev_log.error_once("warning: failed to delete old cython speedups. Please delete all *.so files from the directories "
                       "\"%s\" and \"%s\"" % (_pydevd_bundle_ext_dir, _pydevd_frame_eval_ext_dir))


if USE_CYTHON_FLAG in ENV_TRUE_LOWER_VALUES:
    # We must import the cython version if forcing cython
    from _pydevd_bundle.pydevd_cython_wrapper import trace_dispatch, global_cache_skips, global_cache_frame_skips, fix_top_level_trace_and_get_trace_func
    USING_CYTHON = True

elif USE_CYTHON_FLAG in ENV_FALSE_LOWER_VALUES:
    # Use the regular version if not forcing cython
    from _pydevd_bundle.pydevd_trace_dispatch_regular import trace_dispatch, global_cache_skips, global_cache_frame_skips, fix_top_level_trace_and_get_trace_func  # @UnusedImport

else:
    # Regular: use fallback if not found and give message to user
    try:
        from _pydevd_bundle.pydevd_cython_wrapper import trace_dispatch, global_cache_skips, global_cache_frame_skips, fix_top_level_trace_and_get_trace_func

        # This version number is always available
        from _pydevd_bundle.pydevd_additional_thread_info_regular import version as regular_version
        # This version number from the already compiled cython extension
        from _pydevd_bundle.pydevd_cython_wrapper import version as cython_version
        if cython_version != regular_version:
            # delete_old_compiled_extensions() -- would be ok in dev mode but we don't want to erase
            # files from other python versions on release, so, just raise import error here.
            raise ImportError('Cython version of speedups does not match.')
        else:
            USING_CYTHON = True

    except ImportError:
        from _pydevd_bundle.pydevd_trace_dispatch_regular import trace_dispatch, global_cache_skips, global_cache_frame_skips, fix_top_level_trace_and_get_trace_func  # @UnusedImport
        pydev_log.show_compile_cython_command_line()

