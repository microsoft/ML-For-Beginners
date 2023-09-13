from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log


class LineBreakpointWithLazyValidation(LineBreakpoint):

    def __init__(self, *args, **kwargs):
        LineBreakpoint.__init__(self, *args, **kwargs)
        # This is the _AddBreakpointResult that'll be modified (and then re-sent on the
        # on_changed_breakpoint_state).
        self.add_breakpoint_result = None

        # The signature for the callback should be:
        #     on_changed_breakpoint_state(breakpoint_id: int, add_breakpoint_result: _AddBreakpointResult)
        self.on_changed_breakpoint_state = None

        # When its state is checked (in which case it'd call on_changed_breakpoint_state if the
        # state changed), we store a cache key in 'verified_cache_key' -- in case it changes
        # we'd need to re-verify it (for instance, the template could have changed on disk).
        self.verified_cache_key = None


class ValidationInfo(object):

    def __init__(self):
        self._canonical_normalized_filename_to_last_template_lines = {}

    def _collect_valid_lines_in_template(self, template):
        # We cache the lines in the template itself. Note that among requests the
        # template may be a different instance (because the template contents could be
        # changed on disk), but this may still be called multiple times during the
        # same render session, so, caching is interesting.
        lines_cache = getattr(template, '__pydevd_lines_cache__', None)
        if lines_cache is not None:
            lines, sorted_lines = lines_cache
            return lines, sorted_lines

        lines = self._collect_valid_lines_in_template_uncached(template)

        lines = frozenset(lines)
        sorted_lines = tuple(sorted(lines))
        template.__pydevd_lines_cache__ = lines, sorted_lines
        return lines, sorted_lines

    def _collect_valid_lines_in_template_uncached(self, template):
        raise NotImplementedError()

    def verify_breakpoints(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, template):
        '''
        This function should be called whenever a rendering is detected.

        :param str canonical_normalized_filename:
        :param dict[int:LineBreakpointWithLazyValidation] template_breakpoints_for_file:
        '''
        valid_lines_frozenset, sorted_lines = self._collect_valid_lines_in_template(template)

        self._canonical_normalized_filename_to_last_template_lines[canonical_normalized_filename] = valid_lines_frozenset, sorted_lines
        self._verify_breakpoints_with_lines_collected(py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines)

    def verify_breakpoints_from_template_cached_lines(self, py_db, canonical_normalized_filename, template_breakpoints_for_file):
        '''
        This is used when the lines are already available (if just the template is available,
        `verify_breakpoints` should be used instead).
        '''
        cached = self._canonical_normalized_filename_to_last_template_lines.get(canonical_normalized_filename)
        if cached is not None:
            valid_lines_frozenset, sorted_lines = cached
            self._verify_breakpoints_with_lines_collected(py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines)

    def _verify_breakpoints_with_lines_collected(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines):
        for line, template_bp in list(template_breakpoints_for_file.items()):  # Note: iterate in a copy (we may mutate it).
            if template_bp.verified_cache_key != valid_lines_frozenset:
                template_bp.verified_cache_key = valid_lines_frozenset
                valid = line in valid_lines_frozenset

                if not valid:
                    new_line = -1
                    if sorted_lines:
                        # Adjust to the first preceding valid line.
                        idx = bisect.bisect_left(sorted_lines, line)
                        if idx > 0:
                            new_line = sorted_lines[idx - 1]

                    if new_line >= 0 and new_line not in template_breakpoints_for_file:
                        # We just add it if found and if there's no existing breakpoint at that
                        # location.
                        if template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_NO_ERROR and template_bp.add_breakpoint_result.translated_line != new_line:
                            pydev_log.debug('Template breakpoint in %s in line: %s moved to line: %s', canonical_normalized_filename, line, new_line)
                            template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR
                            template_bp.add_breakpoint_result.translated_line = new_line

                            # Add it to a new line.
                            template_breakpoints_for_file.pop(line, None)
                            template_breakpoints_for_file[new_line] = template_bp
                            template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)
                    else:
                        if template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_INVALID_LINE:
                            pydev_log.debug('Template breakpoint in %s in line: %s invalid (valid lines: %s)', canonical_normalized_filename, line, valid_lines_frozenset)
                            template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_INVALID_LINE
                            template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)
                else:
                    if template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_NO_ERROR:
                        template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR
                        template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)

