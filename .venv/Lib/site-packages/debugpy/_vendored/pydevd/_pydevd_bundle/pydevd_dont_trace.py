'''
Support for a tag that allows skipping over functions while debugging.
'''
import linecache
import re

# To suppress tracing a method, add the tag @DontTrace
# to a comment either preceding or on the same line as
# the method definition
#
# E.g.:
# #@DontTrace
# def test1():
#     pass
#
#  ... or ...
#
# def test2(): #@DontTrace
#     pass
DONT_TRACE_TAG = '@DontTrace'

# Regular expression to match a decorator (at the beginning
# of a line).
RE_DECORATOR = re.compile(r'^\s*@')

# Mapping from code object to bool.
# If the key exists, the value is the cached result of should_trace_hook
_filename_to_ignored_lines = {}


def default_should_trace_hook(frame, absolute_filename):
    '''
    Return True if this frame should be traced, False if tracing should be blocked.
    '''
    # First, check whether this code object has a cached value
    ignored_lines = _filename_to_ignored_lines.get(absolute_filename)
    if ignored_lines is None:
        # Now, look up that line of code and check for a @DontTrace
        # preceding or on the same line as the method.
        # E.g.:
        # #@DontTrace
        # def test():
        #     pass
        #  ... or ...
        # def test(): #@DontTrace
        #     pass
        ignored_lines = {}
        lines = linecache.getlines(absolute_filename)
        for i_line, line in enumerate(lines):
            j = line.find('#')
            if j >= 0:
                comment = line[j:]
                if DONT_TRACE_TAG in comment:
                    ignored_lines[i_line] = 1

                    # Note: when it's found in the comment, mark it up and down for the decorator lines found.
                    k = i_line - 1
                    while k >= 0:
                        if RE_DECORATOR.match(lines[k]):
                            ignored_lines[k] = 1
                            k -= 1
                        else:
                            break

                    k = i_line + 1
                    while k <= len(lines):
                        if RE_DECORATOR.match(lines[k]):
                            ignored_lines[k] = 1
                            k += 1
                        else:
                            break

        _filename_to_ignored_lines[absolute_filename] = ignored_lines

    func_line = frame.f_code.co_firstlineno - 1  # co_firstlineno is 1-based, so -1 is needed
    return not (
        func_line - 1 in ignored_lines or  # -1 to get line before method
        func_line in ignored_lines)  # method line


should_trace_hook = None


def clear_trace_filter_cache():
    '''
    Clear the trace filter cache.
    Call this after reloading.
    '''
    global should_trace_hook
    try:
        # Need to temporarily disable a hook because otherwise
        # _filename_to_ignored_lines.clear() will never complete.
        old_hook = should_trace_hook
        should_trace_hook = None

        # Clear the linecache
        linecache.clearcache()
        _filename_to_ignored_lines.clear()

    finally:
        should_trace_hook = old_hook


def trace_filter(mode):
    '''
    Set the trace filter mode.

    mode: Whether to enable the trace hook.
      True: Trace filtering on (skipping methods tagged @DontTrace)
      False: Trace filtering off (trace methods tagged @DontTrace)
      None/default: Toggle trace filtering.
    '''
    global should_trace_hook
    if mode is None:
        mode = should_trace_hook is None

    if mode:
        should_trace_hook = default_should_trace_hook
    else:
        should_trace_hook = None

    return mode

