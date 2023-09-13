# Defines which version of the PyDBAdditionalThreadInfo we'll use.
from _pydevd_bundle.pydevd_constants import ENV_FALSE_LOWER_VALUES, USE_CYTHON_FLAG, \
    ENV_TRUE_LOWER_VALUES

if USE_CYTHON_FLAG in ENV_TRUE_LOWER_VALUES:
    # We must import the cython version if forcing cython
    from _pydevd_bundle.pydevd_cython_wrapper import PyDBAdditionalThreadInfo, set_additional_thread_info, _set_additional_thread_info_lock  # @UnusedImport

elif USE_CYTHON_FLAG in ENV_FALSE_LOWER_VALUES:
    # Use the regular version if not forcing cython
    from _pydevd_bundle.pydevd_additional_thread_info_regular import PyDBAdditionalThreadInfo, set_additional_thread_info, _set_additional_thread_info_lock  # @UnusedImport @Reimport

else:
    # Regular: use fallback if not found (message is already given elsewhere).
    try:
        from _pydevd_bundle.pydevd_cython_wrapper import PyDBAdditionalThreadInfo, set_additional_thread_info, _set_additional_thread_info_lock
    except ImportError:
        from _pydevd_bundle.pydevd_additional_thread_info_regular import PyDBAdditionalThreadInfo, set_additional_thread_info, _set_additional_thread_info_lock  # @UnusedImport

