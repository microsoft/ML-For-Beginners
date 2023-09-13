'''
Sample usage to track changes in a thread.

    import threading
    import time
    watcher = fsnotify.Watcher()
    watcher.accepted_file_extensions = {'.py', '.pyw'}

    # Configure target values to compute throttling.
    # Note: internal sleep times will be updated based on
    # profiling the actual application runtime to match
    # those values.

    watcher.target_time_for_single_scan = 2.
    watcher.target_time_for_notification = 4.

    watcher.set_tracked_paths([target_dir])

    def start_watching():  # Called from thread
        for change_enum, change_path in watcher.iter_changes():
            if change_enum == fsnotify.Change.added:
                print('Added: ', change_path)
            elif change_enum == fsnotify.Change.modified:
                print('Modified: ', change_path)
            elif change_enum == fsnotify.Change.deleted:
                print('Deleted: ', change_path)

    t = threading.Thread(target=start_watching)
    t.daemon = True
    t.start()

    try:
        ...
    finally:
        watcher.dispose()


Note: changes are only reported for files (added/modified/deleted), not directories.
'''
import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir

try:
    from enum import IntEnum
except:

    class IntEnum(object):
        pass

import time

__author__ = 'Fabio Zadrozny'
__email__ = 'fabiofz@gmail.com'
__version__ = '0.1.5'  # Version here and in setup.py


class Change(IntEnum):
    added = 1
    modified = 2
    deleted = 3


class _SingleVisitInfo(object):

    def __init__(self):
        self.count = 0
        self.visited_dirs = set()
        self.file_to_mtime = {}
        self.last_sleep_time = time.time()


class _PathWatcher(object):
    '''
    Helper to watch a single path.
    '''

    def __init__(self, root_path, accept_directory, accept_file, single_visit_info, max_recursion_level, sleep_time=.0):
        '''
        :type root_path: str
        :type accept_directory: Callback[str, bool]
        :type accept_file: Callback[str, bool]
        :type max_recursion_level: int
        :type sleep_time: float
        '''
        self.accept_directory = accept_directory
        self.accept_file = accept_file
        self._max_recursion_level = max_recursion_level

        self._root_path = root_path

        # Initial sleep value for throttling, it'll be auto-updated based on the
        # Watcher.target_time_for_single_scan.
        self.sleep_time = sleep_time

        self.sleep_at_elapsed = 1. / 30.

        # When created, do the initial snapshot right away!
        old_file_to_mtime = {}
        self._check(single_visit_info, lambda _change: None, old_file_to_mtime)

    def __eq__(self, o):
        if isinstance(o, _PathWatcher):
            return self._root_path == o._root_path

        return False

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash(self._root_path)

    def _check_dir(self, dir_path, single_visit_info, append_change, old_file_to_mtime, level):
        # This is the actual poll loop
        if dir_path in single_visit_info.visited_dirs or level > self._max_recursion_level:
            return
        single_visit_info.visited_dirs.add(dir_path)
        try:
            if isinstance(dir_path, bytes):
                try:
                    dir_path = dir_path.decode(sys.getfilesystemencoding())
                except UnicodeDecodeError:
                    try:
                        dir_path = dir_path.decode('utf-8')
                    except UnicodeDecodeError:
                        return  # Ignore if we can't deal with the path.

            new_files = single_visit_info.file_to_mtime

            for entry in scandir(dir_path):
                single_visit_info.count += 1

                # Throttle if needed inside the loop
                # to avoid consuming too much CPU.
                if single_visit_info.count % 300 == 0:
                    if self.sleep_time > 0:
                        t = time.time()
                        diff = t - single_visit_info.last_sleep_time
                        if diff > self.sleep_at_elapsed:
                            time.sleep(self.sleep_time)
                            single_visit_info.last_sleep_time = time.time()

                if entry.is_dir():
                    if self.accept_directory(entry.path):
                        self._check_dir(entry.path, single_visit_info, append_change, old_file_to_mtime, level + 1)

                elif self.accept_file(entry.path):
                    stat = entry.stat()
                    mtime = (stat.st_mtime_ns, stat.st_size)
                    path = entry.path
                    new_files[path] = mtime

                    old_mtime = old_file_to_mtime.pop(path, None)
                    if not old_mtime:
                        append_change((Change.added, path))
                    elif old_mtime != mtime:
                        append_change((Change.modified, path))

        except OSError:
            pass  # Directory was removed in the meanwhile.

    def _check(self, single_visit_info, append_change, old_file_to_mtime):
        self._check_dir(self._root_path, single_visit_info, append_change, old_file_to_mtime, 0)


class Watcher(object):

    # By default (if accept_directory is not specified), these will be the
    # ignored directories.
    ignored_dirs = {u'.git', u'__pycache__', u'.idea', u'node_modules', u'.metadata'}

    # By default (if accept_file is not specified), these will be the
    # accepted files.
    accepted_file_extensions = ()

    # Set to the target value for doing full scan of all files (adds a sleep inside the poll loop
    # which processes files to reach the target time).
    # Lower values will consume more CPU
    # Set to 0.0 to have no sleeps (which will result in a higher cpu load).
    target_time_for_single_scan = 2.0

    # Set the target value from the start of one scan to the start of another scan (adds a
    # sleep after a full poll is done to reach the target time).
    # Lower values will consume more CPU.
    # Set to 0.0 to have a new scan start right away without any sleeps.
    target_time_for_notification = 4.0

    # Set to True to print the time for a single poll through all the paths.
    print_poll_time = False

    # This is the maximum recursion level.
    max_recursion_level = 10

    def __init__(self, accept_directory=None, accept_file=None):
        '''
        :param Callable[str, bool] accept_directory:
            Callable that returns whether a directory should be watched.
            Note: if passed it'll override the `ignored_dirs`

        :param Callable[str, bool] accept_file:
            Callable that returns whether a file should be watched.
            Note: if passed it'll override the `accepted_file_extensions`.
        '''
        self._path_watchers = set()
        self._disposed = threading.Event()

        if accept_directory is None:
            accept_directory = lambda dir_path: basename(dir_path) not in self.ignored_dirs
        if accept_file is None:
            accept_file = lambda path_name: \
                not self.accepted_file_extensions or path_name.endswith(self.accepted_file_extensions)
        self.accept_file = accept_file
        self.accept_directory = accept_directory
        self._single_visit_info = _SingleVisitInfo()

    @property
    def accept_directory(self):
        return self._accept_directory

    @accept_directory.setter
    def accept_directory(self, accept_directory):
        self._accept_directory = accept_directory
        for path_watcher in self._path_watchers:
            path_watcher.accept_directory = accept_directory

    @property
    def accept_file(self):
        return self._accept_file

    @accept_file.setter
    def accept_file(self, accept_file):
        self._accept_file = accept_file
        for path_watcher in self._path_watchers:
            path_watcher.accept_file = accept_file

    def dispose(self):
        self._disposed.set()

    @property
    def path_watchers(self):
        return tuple(self._path_watchers)

    def set_tracked_paths(self, paths):
        """
        Note: always resets all path trackers to track the passed paths.
        """
        if not isinstance(paths, (list, tuple, set)):
            paths = (paths,)

        # Sort by the path len so that the bigger paths come first (so,
        # if there's any nesting we want the nested paths to be visited
        # before the parent paths so that the max_recursion_level is correct).
        paths = sorted(set(paths), key=lambda path:-len(path))
        path_watchers = set()

        self._single_visit_info = _SingleVisitInfo()

        initial_time = time.time()
        for path in paths:
            sleep_time = 0.  # When collecting the first time, sleep_time should be 0!
            path_watcher = _PathWatcher(
                path,
                self.accept_directory,
                self.accept_file,
                self._single_visit_info,
                max_recursion_level=self.max_recursion_level,
                sleep_time=sleep_time,
            )

            path_watchers.add(path_watcher)

        actual_time = (time.time() - initial_time)

        pydev_log.debug('Tracking the following paths for changes: %s', paths)
        pydev_log.debug('Time to track: %.2fs', actual_time)
        pydev_log.debug('Folders found: %s', len(self._single_visit_info.visited_dirs))
        pydev_log.debug('Files found: %s', len(self._single_visit_info.file_to_mtime))
        self._path_watchers = path_watchers

    def iter_changes(self):
        '''
        Continuously provides changes (until dispose() is called).

        Changes provided are tuples with the Change enum and filesystem path.

        :rtype: Iterable[Tuple[Change, str]]
        '''
        while not self._disposed.is_set():
            initial_time = time.time()

            old_visit_info = self._single_visit_info
            old_file_to_mtime = old_visit_info.file_to_mtime
            changes = []
            append_change = changes.append

            self._single_visit_info = single_visit_info = _SingleVisitInfo()
            for path_watcher in self._path_watchers:
                path_watcher._check(single_visit_info, append_change, old_file_to_mtime)

            # Note that we pop entries while visiting, so, what remained is what's deleted.
            for entry in old_file_to_mtime:
                append_change((Change.deleted, entry))

            for change in changes:
                yield change

            actual_time = (time.time() - initial_time)
            if self.print_poll_time:
                print('--- Total poll time: %.3fs' % actual_time)

            if actual_time > 0:
                if self.target_time_for_single_scan <= 0.0:
                    for path_watcher in self._path_watchers:
                        path_watcher.sleep_time = 0.0
                else:
                    perc = self.target_time_for_single_scan / actual_time

                    # Prevent from changing the values too much (go slowly into the right
                    # direction).
                    # (to prevent from cases where the user puts the machine on sleep and
                    # values become too skewed).
                    if perc > 2.:
                        perc = 2.
                    elif perc < 0.5:
                        perc = 0.5

                    for path_watcher in self._path_watchers:
                        if path_watcher.sleep_time <= 0.0:
                            path_watcher.sleep_time = 0.001
                        new_sleep_time = path_watcher.sleep_time * perc

                        # Prevent from changing the values too much (go slowly into the right
                        # direction).
                        # (to prevent from cases where the user puts the machine on sleep and
                        # values become too skewed).
                        diff_sleep_time = new_sleep_time - path_watcher.sleep_time
                        path_watcher.sleep_time += (diff_sleep_time / (3.0 * len(self._path_watchers)))

                        if actual_time > 0:
                            self._disposed.wait(actual_time)

                        if path_watcher.sleep_time < 0.001:
                            path_watcher.sleep_time = 0.001

            # print('new sleep time: %s' % path_watcher.sleep_time)

            diff = self.target_time_for_notification - actual_time
            if diff > 0.:
                self._disposed.wait(diff)

