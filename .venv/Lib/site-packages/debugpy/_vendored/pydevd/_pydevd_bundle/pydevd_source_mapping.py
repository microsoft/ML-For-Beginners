import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils


class SourceMappingEntry(object):

    __slots__ = ['source_filename', 'line', 'end_line', 'runtime_line', 'runtime_source']

    def __init__(self, line, end_line, runtime_line, runtime_source):
        assert isinstance(runtime_source, str)

        self.line = int(line)
        self.end_line = int(end_line)
        self.runtime_line = int(runtime_line)
        self.runtime_source = runtime_source  # Something as <ipython-cell-xxx>

        # Should be set after translated to server (absolute_source_filename).
        # This is what's sent to the client afterwards (so, its case should not be normalized).
        self.source_filename = None

    def contains_line(self, i):
        return self.line <= i <= self.end_line

    def contains_runtime_line(self, i):
        line_count = self.end_line + self.line
        runtime_end_line = self.runtime_line + line_count
        return self.runtime_line <= i <= runtime_end_line

    def __str__(self):
        return 'SourceMappingEntry(%s)' % (
            ', '.join('%s=%r' % (attr, getattr(self, attr)) for attr in self.__slots__))

    __repr__ = __str__


class SourceMapping(object):

    def __init__(self, on_source_mapping_changed=NULL):
        self._mappings_to_server = {}  # dict(normalized(file.py) to [SourceMappingEntry])
        self._mappings_to_client = {}  # dict(<cell> to File.py)
        self._cache = {}
        self._on_source_mapping_changed = on_source_mapping_changed

    def set_source_mapping(self, absolute_filename, mapping):
        '''
        :param str absolute_filename:
            The filename for the source mapping (bytes on py2 and str on py3).

        :param list(SourceMappingEntry) mapping:
            A list with the source mapping entries to be applied to the given filename.

        :return str:
            An error message if it was not possible to set the mapping or an empty string if
            everything is ok.
        '''
        # Let's first validate if it's ok to apply that mapping.
        # File mappings must be 1:N, not M:N (i.e.: if there's a mapping from file1.py to <cell1>,
        # there can be no other mapping from any other file to <cell1>).
        # This is a limitation to make it easier to remove existing breakpoints when new breakpoints are
        # set to a file (so, any file matching that breakpoint can be removed instead of needing to check
        # which lines are corresponding to that file).
        for map_entry in mapping:
            existing_source_filename = self._mappings_to_client.get(map_entry.runtime_source)
            if existing_source_filename and existing_source_filename != absolute_filename:
                return 'Cannot apply mapping from %s to %s (it conflicts with mapping: %s to %s)' % (
                    absolute_filename, map_entry.runtime_source, existing_source_filename, map_entry.runtime_source)

        try:
            absolute_normalized_filename = pydevd_file_utils.normcase(absolute_filename)
            current_mapping = self._mappings_to_server.get(absolute_normalized_filename, [])
            for map_entry in current_mapping:
                del self._mappings_to_client[map_entry.runtime_source]

            self._mappings_to_server[absolute_normalized_filename] = sorted(mapping, key=lambda entry:entry.line)

            for map_entry in mapping:
                self._mappings_to_client[map_entry.runtime_source] = absolute_filename
        finally:
            self._cache.clear()
            self._on_source_mapping_changed()
        return ''

    def map_to_client(self, runtime_source_filename, lineno):
        key = (lineno, 'client', runtime_source_filename)
        try:
            return self._cache[key]
        except KeyError:
            for _, mapping in list(self._mappings_to_server.items()):
                for map_entry in mapping:
                    if map_entry.runtime_source == runtime_source_filename:  # <cell1>
                        if map_entry.contains_runtime_line(lineno):  # matches line range
                            self._cache[key] = (map_entry.source_filename, map_entry.line + (lineno - map_entry.runtime_line), True)
                            return self._cache[key]

            self._cache[key] = (runtime_source_filename, lineno, False)  # Mark that no translation happened in the cache.
            return self._cache[key]

    def has_mapping_entry(self, runtime_source_filename):
        '''
        :param runtime_source_filename:
            Something as <ipython-cell-xxx>
        '''
        # Note that we're not interested in the line here, just on knowing if a given filename
        # (from the server) has a mapping for it.
        key = ('has_entry', runtime_source_filename)
        try:
            return self._cache[key]
        except KeyError:
            for _absolute_normalized_filename, mapping in list(self._mappings_to_server.items()):
                for map_entry in mapping:
                    if map_entry.runtime_source == runtime_source_filename:
                        self._cache[key] = True
                        return self._cache[key]

            self._cache[key] = False
            return self._cache[key]

    def map_to_server(self, absolute_filename, lineno):
        '''
        Convert something as 'file1.py' at line 10 to '<ipython-cell-xxx>' at line 2.

        Note that the name should be already normalized at this point.
        '''
        absolute_normalized_filename = pydevd_file_utils.normcase(absolute_filename)

        changed = False
        mappings = self._mappings_to_server.get(absolute_normalized_filename)
        if mappings:

            i = bisect.bisect(KeyifyList(mappings, lambda entry:entry.line), lineno)
            if i >= len(mappings):
                i -= 1

            if i == 0:
                entry = mappings[i]

            else:
                entry = mappings[i - 1]

            if not entry.contains_line(lineno):
                entry = mappings[i]
                if not entry.contains_line(lineno):
                    entry = None

            if entry is not None:
                lineno = entry.runtime_line + (lineno - entry.line)

                absolute_filename = entry.runtime_source
                changed = True

        return absolute_filename, lineno, changed

