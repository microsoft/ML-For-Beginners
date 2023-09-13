""" History related magics and functionality """

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import atexit
import datetime
from pathlib import Path
import re
import sqlite3
import threading

from traitlets.config.configurable import LoggingConfigurable
from decorator import decorator
from IPython.utils.decorators import undoc
from IPython.paths import locate_profile
from traitlets import (
    Any,
    Bool,
    Dict,
    Instance,
    Integer,
    List,
    Unicode,
    Union,
    TraitError,
    default,
    observe,
)

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

@undoc
class DummyDB(object):
    """Dummy DB that will act as a black hole for history.

    Only used in the absence of sqlite"""
    def execute(*args, **kwargs):
        return []

    def commit(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


@decorator
def only_when_enabled(f, self, *a, **kw):
    """Decorator: return an empty list in the absence of sqlite."""
    if not self.enabled:
        return []
    else:
        return f(self, *a, **kw)


# use 16kB as threshold for whether a corrupt history db should be saved
# that should be at least 100 entries or so
_SAVE_DB_SIZE = 16384

@decorator
def catch_corrupt_db(f, self, *a, **kw):
    """A decorator which wraps HistoryAccessor method calls to catch errors from
    a corrupt SQLite database, move the old database out of the way, and create
    a new one.

    We avoid clobbering larger databases because this may be triggered due to filesystem issues,
    not just a corrupt file.
    """
    try:
        return f(self, *a, **kw)
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        self._corrupt_db_counter += 1
        self.log.error("Failed to open SQLite history %s (%s).", self.hist_file, e)
        if self.hist_file != ':memory:':
            if self._corrupt_db_counter > self._corrupt_db_limit:
                self.hist_file = ':memory:'
                self.log.error("Failed to load history too many times, history will not be saved.")
            elif self.hist_file.is_file():
                # move the file out of the way
                base = str(self.hist_file.parent / self.hist_file.stem)
                ext = self.hist_file.suffix
                size = self.hist_file.stat().st_size
                if size >= _SAVE_DB_SIZE:
                    # if there's significant content, avoid clobbering
                    now = datetime.datetime.now().isoformat().replace(':', '.')
                    newpath = base + '-corrupt-' + now + ext
                    # don't clobber previous corrupt backups
                    for i in range(100):
                        if not Path(newpath).exists():
                            break
                        else:
                            newpath = base + '-corrupt-' + now + (u'-%i' % i) + ext
                else:
                    # not much content, possibly empty; don't worry about clobbering
                    # maybe we should just delete it?
                    newpath = base + '-corrupt' + ext
                self.hist_file.rename(newpath)
                self.log.error("History file was moved to %s and a new file created.", newpath)
            self.init_db()
            return []
        else:
            # Failed with :memory:, something serious is wrong
            raise


class HistoryAccessorBase(LoggingConfigurable):
    """An abstract class for History Accessors """

    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        raise NotImplementedError

    def search(self, pattern="*", raw=True, search_raw=True,
               output=False, n=None, unique=False):
        raise NotImplementedError

    def get_range(self, session, start=1, stop=None, raw=True,output=False):
        raise NotImplementedError

    def get_range_by_str(self, rangestr, raw=True, output=False):
        raise NotImplementedError


class HistoryAccessor(HistoryAccessorBase):
    """Access the history database without adding to it.

    This is intended for use by standalone history tools. IPython shells use
    HistoryManager, below, which is a subclass of this."""

    # counter for init_db retries, so we don't keep trying over and over
    _corrupt_db_counter = 0
     # after two failures, fallback on :memory:
    _corrupt_db_limit = 2

    # String holding the path to the history file
    hist_file = Union(
        [Instance(Path), Unicode()],
        help="""Path to file to use for SQLite history database.

        By default, IPython will put the history database in the IPython
        profile directory.  If you would rather share one history among
        profiles, you can set this value in each, so that they are consistent.

        Due to an issue with fcntl, SQLite is known to misbehave on some NFS
        mounts.  If you see IPython hanging, try setting this to something on a
        local disk, e.g::

            ipython --HistoryManager.hist_file=/tmp/ipython_hist.sqlite

        you can also use the specific value `:memory:` (including the colon
        at both end but not the back ticks), to avoid creating an history file.

        """,
    ).tag(config=True)

    enabled = Bool(True,
        help="""enable the SQLite history

        set enabled=False to disable the SQLite history,
        in which case there will be no stored history, no SQLite connection,
        and no background saving thread.  This may be necessary in some
        threaded environments where IPython is embedded.
        """,
    ).tag(config=True)

    connection_options = Dict(
        help="""Options for configuring the SQLite connection

        These options are passed as keyword args to sqlite3.connect
        when establishing database connections.
        """
    ).tag(config=True)

    @default("connection_options")
    def _default_connection_options(self):
        return dict(check_same_thread=False)

    # The SQLite database
    db = Any()
    @observe('db')
    def _db_changed(self, change):
        """validate the db, since it can be an Instance of two different types"""
        new = change['new']
        connection_types = (DummyDB, sqlite3.Connection)
        if not isinstance(new, connection_types):
            msg = "%s.db must be sqlite3 Connection or DummyDB, not %r" % \
                    (self.__class__.__name__, new)
            raise TraitError(msg)

    def __init__(self, profile="default", hist_file="", **traits):
        """Create a new history accessor.

        Parameters
        ----------
        profile : str
            The name of the profile from which to open history.
        hist_file : str
            Path to an SQLite history database stored by IPython. If specified,
            hist_file overrides profile.
        config : :class:`~traitlets.config.loader.Config`
            Config object. hist_file can also be set through this.
        """
        super(HistoryAccessor, self).__init__(**traits)
        # defer setting hist_file from kwarg until after init,
        # otherwise the default kwarg value would clobber any value
        # set by config
        if hist_file:
            self.hist_file = hist_file

        try:
            self.hist_file
        except TraitError:
            # No one has set the hist_file, yet.
            self.hist_file = self._get_hist_file_name(profile)

        self.init_db()

    def _get_hist_file_name(self, profile='default'):
        """Find the history file for the given profile name.

        This is overridden by the HistoryManager subclass, to use the shell's
        active profile.

        Parameters
        ----------
        profile : str
            The name of a profile which has a history file.
        """
        return Path(locate_profile(profile)) / "history.sqlite"

    @catch_corrupt_db
    def init_db(self):
        """Connect to the database, and create tables if necessary."""
        if not self.enabled:
            self.db = DummyDB()
            return

        # use detect_types so that timestamps return datetime objects
        kwargs = dict(detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
        kwargs.update(self.connection_options)
        self.db = sqlite3.connect(str(self.hist_file), **kwargs)
        with self.db:
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (session integer
                            primary key autoincrement, start timestamp,
                            end timestamp, num_cmds integer, remark text)"""
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS history
                    (session integer, line integer, source text, source_raw text,
                    PRIMARY KEY (session, line))"""
            )
            # Output history is optional, but ensure the table's there so it can be
            # enabled later.
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS output_history
                            (session integer, line integer, output text,
                            PRIMARY KEY (session, line))"""
            )
        # success! reset corrupt db count
        self._corrupt_db_counter = 0

    def writeout_cache(self):
        """Overridden by HistoryManager to dump the cache before certain
        database lookups."""
        pass

    ## -------------------------------
    ## Methods for retrieving history:
    ## -------------------------------
    def _run_sql(self, sql, params, raw=True, output=False, latest=False):
        """Prepares and runs an SQL query for the history database.

        Parameters
        ----------
        sql : str
            Any filtering expressions to go after SELECT ... FROM ...
        params : tuple
            Parameters passed to the SQL query (to replace "?")
        raw, output : bool
            See :meth:`get_range`
        latest : bool
            Select rows with max (session, line)

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        toget = 'source_raw' if raw else 'source'
        sqlfrom = "history"
        if output:
            sqlfrom = "history LEFT JOIN output_history USING (session, line)"
            toget = "history.%s, output_history.output" % toget
        if latest:
            toget += ", MAX(session * 128 * 1024 + line)"
        this_querry = "SELECT session, line, %s FROM %s " % (toget, sqlfrom) + sql
        cur = self.db.execute(this_querry, params)
        if latest:
            cur = (row[:-1] for row in cur)
        if output:    # Regroup into 3-tuples, and parse JSON
            return ((ses, lin, (inp, out)) for ses, lin, inp, out in cur)
        return cur

    @only_when_enabled
    @catch_corrupt_db
    def get_session_info(self, session):
        """Get info about a session.

        Parameters
        ----------
        session : int
            Session number to retrieve.

        Returns
        -------
        session_id : int
            Session ID number
        start : datetime
            Timestamp for the start of the session.
        end : datetime
            Timestamp for the end of the session, or None if IPython crashed.
        num_cmds : int
            Number of commands run, or None if IPython crashed.
        remark : unicode
            A manually set description.
        """
        query = "SELECT * from sessions where session == ?"
        return self.db.execute(query, (session,)).fetchone()

    @catch_corrupt_db
    def get_last_session_id(self):
        """Get the last session ID currently in the database.

        Within IPython, this should be the same as the value stored in
        :attr:`HistoryManager.session_number`.
        """
        for record in self.get_tail(n=1, include_latest=True):
            return record[0]

    @catch_corrupt_db
    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        """Get the last n lines from the history database.

        Parameters
        ----------
        n : int
            The number of lines to get
        raw, output : bool
            See :meth:`get_range`
        include_latest : bool
            If False (default), n+1 lines are fetched, and the latest one
            is discarded. This is intended to be used where the function
            is called by a user command, which it should not return.

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        self.writeout_cache()
        if not include_latest:
            n += 1
        cur = self._run_sql(
            "ORDER BY session DESC, line DESC LIMIT ?", (n,), raw=raw, output=output
        )
        if not include_latest:
            return reversed(list(cur)[1:])
        return reversed(list(cur))

    @catch_corrupt_db
    def search(self, pattern="*", raw=True, search_raw=True,
               output=False, n=None, unique=False):
        """Search the database using unix glob-style matching (wildcards
        * and ?).

        Parameters
        ----------
        pattern : str
            The wildcarded pattern to match when searching
        search_raw : bool
            If True, search the raw input, otherwise, the parsed input
        raw, output : bool
            See :meth:`get_range`
        n : None or int
            If an integer is given, it defines the limit of
            returned entries.
        unique : bool
            When it is true, return only unique entries.

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        tosearch = "source_raw" if search_raw else "source"
        if output:
            tosearch = "history." + tosearch
        self.writeout_cache()
        sqlform = "WHERE %s GLOB ?" % tosearch
        params = (pattern,)
        if unique:
            sqlform += ' GROUP BY {0}'.format(tosearch)
        if n is not None:
            sqlform += " ORDER BY session DESC, line DESC LIMIT ?"
            params += (n,)
        elif unique:
            sqlform += " ORDER BY session, line"
        cur = self._run_sql(sqlform, params, raw=raw, output=output, latest=unique)
        if n is not None:
            return reversed(list(cur))
        return cur

    @catch_corrupt_db
    def get_range(self, session, start=1, stop=None, raw=True,output=False):
        """Retrieve input by session.

        Parameters
        ----------
        session : int
            Session number to retrieve.
        start : int
            First line to retrieve.
        stop : int
            End of line range (excluded from output itself). If None, retrieve
            to the end of the session.
        raw : bool
            If True, return untranslated input
        output : bool
            If True, attempt to include output. This will be 'real' Python
            objects for the current session, or text reprs from previous
            sessions if db_log_output was enabled at the time. Where no output
            is found, None is used.

        Returns
        -------
        entries
            An iterator over the desired lines. Each line is a 3-tuple, either
            (session, line, input) if output is False, or
            (session, line, (input, output)) if output is True.
        """
        if stop:
            lineclause = "line >= ? AND line < ?"
            params = (session, start, stop)
        else:
            lineclause = "line>=?"
            params = (session, start)

        return self._run_sql("WHERE session==? AND %s" % lineclause,
                                    params, raw=raw, output=output)

    def get_range_by_str(self, rangestr, raw=True, output=False):
        """Get lines of history from a string of ranges, as used by magic
        commands %hist, %save, %macro, etc.

        Parameters
        ----------
        rangestr : str
            A string specifying ranges, e.g. "5 ~2/1-4". If empty string is used,
            this will return everything from current session's history.

            See the documentation of :func:`%history` for the full details.

        raw, output : bool
            As :meth:`get_range`

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        for sess, s, e in extract_hist_ranges(rangestr):
            for line in self.get_range(sess, s, e, raw=raw, output=output):
                yield line


class HistoryManager(HistoryAccessor):
    """A class to organize all history-related functionality in one place.
    """
    # Public interface

    # An instance of the IPython shell we are attached to
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC',
                     allow_none=True)
    # Lists to hold processed and raw history. These start with a blank entry
    # so that we can index them starting from 1
    input_hist_parsed = List([""])
    input_hist_raw = List([""])
    # A list of directories visited during session
    dir_hist = List()
    @default('dir_hist')
    def _dir_hist_default(self):
        try:
            return [Path.cwd()]
        except OSError:
            return []

    # A dict of output history, keyed with ints from the shell's
    # execution count.
    output_hist = Dict()
    # The text/plain repr of outputs.
    output_hist_reprs = Dict()

    # The number of the current session in the history database
    session_number = Integer()

    db_log_output = Bool(False,
        help="Should the history database include output? (default: no)"
    ).tag(config=True)
    db_cache_size = Integer(0,
        help="Write to database every x commands (higher values save disk access & power).\n"
        "Values of 1 or less effectively disable caching."
    ).tag(config=True)
    # The input and output caches
    db_input_cache = List()
    db_output_cache = List()

    # History saving in separate thread
    save_thread = Instance('IPython.core.history.HistorySavingThread',
                           allow_none=True)
    save_flag = Instance(threading.Event, allow_none=True)

    # Private interface
    # Variables used to store the three last inputs from the user.  On each new
    # history update, we populate the user's namespace with these, shifted as
    # necessary.
    _i00 = Unicode(u'')
    _i = Unicode(u'')
    _ii = Unicode(u'')
    _iii = Unicode(u'')

    # A regex matching all forms of the exit command, so that we don't store
    # them in the history (it's annoying to rewind the first entry and land on
    # an exit call).
    _exit_re = re.compile(r"(exit|quit)(\s*\(.*\))?$")

    def __init__(self, shell=None, config=None, **traits):
        """Create a new history manager associated with a shell instance.
        """
        super(HistoryManager, self).__init__(shell=shell, config=config,
            **traits)
        self.save_flag = threading.Event()
        self.db_input_cache_lock = threading.Lock()
        self.db_output_cache_lock = threading.Lock()

        try:
            self.new_session()
        except sqlite3.OperationalError:
            self.log.error("Failed to create history session in %s. History will not be saved.",
                self.hist_file, exc_info=True)
            self.hist_file = ':memory:'

        if self.enabled and self.hist_file != ':memory:':
            self.save_thread = HistorySavingThread(self)
            self.save_thread.start()

    def _get_hist_file_name(self, profile=None):
        """Get default history file name based on the Shell's profile.

        The profile parameter is ignored, but must exist for compatibility with
        the parent class."""
        profile_dir = self.shell.profile_dir.location
        return Path(profile_dir) / "history.sqlite"

    @only_when_enabled
    def new_session(self, conn=None):
        """Get a new session number."""
        if conn is None:
            conn = self.db

        with conn:
            cur = conn.execute(
                """INSERT INTO sessions VALUES (NULL, ?, NULL,
                            NULL, '') """,
                (datetime.datetime.now().isoformat(" "),),
            )
            self.session_number = cur.lastrowid

    def end_session(self):
        """Close the database session, filling in the end time and line count."""
        self.writeout_cache()
        with self.db:
            self.db.execute(
                """UPDATE sessions SET end=?, num_cmds=? WHERE
                            session==?""",
                (
                    datetime.datetime.now().isoformat(" "),
                    len(self.input_hist_parsed) - 1,
                    self.session_number,
                ),
            )
        self.session_number = 0

    def name_session(self, name):
        """Give the current session a name in the history database."""
        with self.db:
            self.db.execute("UPDATE sessions SET remark=? WHERE session==?",
                            (name, self.session_number))

    def reset(self, new_session=True):
        """Clear the session history, releasing all object references, and
        optionally open a new session."""
        self.output_hist.clear()
        # The directory history can't be completely empty
        self.dir_hist[:] = [Path.cwd()]

        if new_session:
            if self.session_number:
                self.end_session()
            self.input_hist_parsed[:] = [""]
            self.input_hist_raw[:] = [""]
            self.new_session()

    # ------------------------------
    # Methods for retrieving history
    # ------------------------------
    def get_session_info(self, session=0):
        """Get info about a session.

        Parameters
        ----------
        session : int
            Session number to retrieve. The current session is 0, and negative
            numbers count back from current session, so -1 is the previous session.

        Returns
        -------
        session_id : int
            Session ID number
        start : datetime
            Timestamp for the start of the session.
        end : datetime
            Timestamp for the end of the session, or None if IPython crashed.
        num_cmds : int
            Number of commands run, or None if IPython crashed.
        remark : unicode
            A manually set description.
        """
        if session <= 0:
            session += self.session_number

        return super(HistoryManager, self).get_session_info(session=session)

    @catch_corrupt_db
    def get_tail(self, n=10, raw=True, output=False, include_latest=False):
        """Get the last n lines from the history database.

        Most recent entry last.

        Completion will be reordered so that that the last ones are when
        possible from current session.

        Parameters
        ----------
        n : int
            The number of lines to get
        raw, output : bool
            See :meth:`get_range`
        include_latest : bool
            If False (default), n+1 lines are fetched, and the latest one
            is discarded. This is intended to be used where the function
            is called by a user command, which it should not return.

        Returns
        -------
        Tuples as :meth:`get_range`
        """
        self.writeout_cache()
        if not include_latest:
            n += 1
        # cursor/line/entry
        this_cur = list(
            self._run_sql(
                "WHERE session == ? ORDER BY line DESC LIMIT ?  ",
                (self.session_number, n),
                raw=raw,
                output=output,
            )
        )
        other_cur = list(
            self._run_sql(
                "WHERE session != ? ORDER BY session DESC, line DESC LIMIT ?",
                (self.session_number, n),
                raw=raw,
                output=output,
            )
        )

        everything = this_cur + other_cur

        everything = everything[:n]

        if not include_latest:
            return list(everything)[:0:-1]
        return list(everything)[::-1]

    def _get_range_session(self, start=1, stop=None, raw=True, output=False):
        """Get input and output history from the current session. Called by
        get_range, and takes similar parameters."""
        input_hist = self.input_hist_raw if raw else self.input_hist_parsed

        n = len(input_hist)
        if start < 0:
            start += n
        if not stop or (stop > n):
            stop = n
        elif stop < 0:
            stop += n

        for i in range(start, stop):
            if output:
                line = (input_hist[i], self.output_hist_reprs.get(i))
            else:
                line = input_hist[i]
            yield (0, i, line)

    def get_range(self, session=0, start=1, stop=None, raw=True,output=False):
        """Retrieve input by session.

        Parameters
        ----------
        session : int
            Session number to retrieve. The current session is 0, and negative
            numbers count back from current session, so -1 is previous session.
        start : int
            First line to retrieve.
        stop : int
            End of line range (excluded from output itself). If None, retrieve
            to the end of the session.
        raw : bool
            If True, return untranslated input
        output : bool
            If True, attempt to include output. This will be 'real' Python
            objects for the current session, or text reprs from previous
            sessions if db_log_output was enabled at the time. Where no output
            is found, None is used.

        Returns
        -------
        entries
            An iterator over the desired lines. Each line is a 3-tuple, either
            (session, line, input) if output is False, or
            (session, line, (input, output)) if output is True.
        """
        if session <= 0:
            session += self.session_number
        if session==self.session_number:          # Current session
            return self._get_range_session(start, stop, raw, output)
        return super(HistoryManager, self).get_range(session, start, stop, raw,
                                                     output)

    ## ----------------------------
    ## Methods for storing history:
    ## ----------------------------
    def store_inputs(self, line_num, source, source_raw=None):
        """Store source and raw input in history and create input cache
        variables ``_i*``.

        Parameters
        ----------
        line_num : int
            The prompt number of this input.
        source : str
            Python input.
        source_raw : str, optional
            If given, this is the raw input without any IPython transformations
            applied to it.  If not given, ``source`` is used.
        """
        if source_raw is None:
            source_raw = source
        source = source.rstrip('\n')
        source_raw = source_raw.rstrip('\n')

        # do not store exit/quit commands
        if self._exit_re.match(source_raw.strip()):
            return

        self.input_hist_parsed.append(source)
        self.input_hist_raw.append(source_raw)

        with self.db_input_cache_lock:
            self.db_input_cache.append((line_num, source, source_raw))
            # Trigger to flush cache and write to DB.
            if len(self.db_input_cache) >= self.db_cache_size:
                self.save_flag.set()

        # update the auto _i variables
        self._iii = self._ii
        self._ii = self._i
        self._i = self._i00
        self._i00 = source_raw

        # hackish access to user namespace to create _i1,_i2... dynamically
        new_i = '_i%s' % line_num
        to_main = {'_i': self._i,
                   '_ii': self._ii,
                   '_iii': self._iii,
                   new_i : self._i00 }

        if self.shell is not None:
            self.shell.push(to_main, interactive=False)

    def store_output(self, line_num):
        """If database output logging is enabled, this saves all the
        outputs from the indicated prompt number to the database. It's
        called by run_cell after code has been executed.

        Parameters
        ----------
        line_num : int
            The line number from which to save outputs
        """
        if (not self.db_log_output) or (line_num not in self.output_hist_reprs):
            return
        output = self.output_hist_reprs[line_num]

        with self.db_output_cache_lock:
            self.db_output_cache.append((line_num, output))
        if self.db_cache_size <= 1:
            self.save_flag.set()

    def _writeout_input_cache(self, conn):
        with conn:
            for line in self.db_input_cache:
                conn.execute("INSERT INTO history VALUES (?, ?, ?, ?)",
                                (self.session_number,)+line)

    def _writeout_output_cache(self, conn):
        with conn:
            for line in self.db_output_cache:
                conn.execute("INSERT INTO output_history VALUES (?, ?, ?)",
                                (self.session_number,)+line)

    @only_when_enabled
    def writeout_cache(self, conn=None):
        """Write any entries in the cache to the database."""
        if conn is None:
            conn = self.db

        with self.db_input_cache_lock:
            try:
                self._writeout_input_cache(conn)
            except sqlite3.IntegrityError:
                self.new_session(conn)
                print("ERROR! Session/line number was not unique in",
                      "database. History logging moved to new session",
                                                self.session_number)
                try:
                    # Try writing to the new session. If this fails, don't
                    # recurse
                    self._writeout_input_cache(conn)
                except sqlite3.IntegrityError:
                    pass
            finally:
                self.db_input_cache = []

        with self.db_output_cache_lock:
            try:
                self._writeout_output_cache(conn)
            except sqlite3.IntegrityError:
                print("!! Session/line number for output was not unique",
                      "in database. Output will not be stored.")
            finally:
                self.db_output_cache = []


class HistorySavingThread(threading.Thread):
    """This thread takes care of writing history to the database, so that
    the UI isn't held up while that happens.

    It waits for the HistoryManager's save_flag to be set, then writes out
    the history cache. The main thread is responsible for setting the flag when
    the cache size reaches a defined threshold."""
    daemon = True
    stop_now = False
    enabled = True
    def __init__(self, history_manager):
        super(HistorySavingThread, self).__init__(name="IPythonHistorySavingThread")
        self.history_manager = history_manager
        self.enabled = history_manager.enabled
        atexit.register(self.stop)

    @only_when_enabled
    def run(self):
        # We need a separate db connection per thread:
        try:
            self.db = sqlite3.connect(
                str(self.history_manager.hist_file),
                **self.history_manager.connection_options,
            )
            while True:
                self.history_manager.save_flag.wait()
                if self.stop_now:
                    self.db.close()
                    return
                self.history_manager.save_flag.clear()
                self.history_manager.writeout_cache(self.db)
        except Exception as e:
            print(("The history saving thread hit an unexpected error (%s)."
                   "History will not be written to the database.") % repr(e))

    def stop(self):
        """This can be called from the main thread to safely stop this thread.

        Note that it does not attempt to write out remaining history before
        exiting. That should be done by calling the HistoryManager's
        end_session method."""
        self.stop_now = True
        self.history_manager.save_flag.set()
        self.join()


# To match, e.g. ~5/8-~2/3
range_re = re.compile(r"""
((?P<startsess>~?\d+)/)?
(?P<start>\d+)?
((?P<sep>[\-:])
 ((?P<endsess>~?\d+)/)?
 (?P<end>\d+))?
$""", re.VERBOSE)


def extract_hist_ranges(ranges_str):
    """Turn a string of history ranges into 3-tuples of (session, start, stop).

    Empty string results in a `[(0, 1, None)]`, i.e. "everything from current
    session".

    Examples
    --------
    >>> list(extract_hist_ranges("~8/5-~7/4 2"))
    [(-8, 5, None), (-7, 1, 5), (0, 2, 3)]
    """
    if ranges_str == "":
        yield (0, 1, None)  # Everything from current session
        return

    for range_str in ranges_str.split():
        rmatch = range_re.match(range_str)
        if not rmatch:
            continue
        start = rmatch.group("start")
        if start:
            start = int(start)
            end = rmatch.group("end")
            # If no end specified, get (a, a + 1)
            end = int(end) if end else start + 1
        else:  # start not specified
            if not rmatch.group('startsess'):  # no startsess
                continue
            start = 1
            end = None  # provide the entire session hist

        if rmatch.group("sep") == "-":       # 1-3 == 1:4 --> [1, 2, 3]
            end += 1
        startsess = rmatch.group("startsess") or "0"
        endsess = rmatch.group("endsess") or startsess
        startsess = int(startsess.replace("~","-"))
        endsess = int(endsess.replace("~","-"))
        assert endsess >= startsess, "start session must be earlier than end session"

        if endsess == startsess:
            yield (startsess, start, end)
            continue
        # Multiple sessions in one range:
        yield (startsess, start, None)
        for sess in range(startsess+1, endsess):
            yield (sess, 1, None)
        yield (endsess, 1, end)


def _format_lineno(session, line):
    """Helper function to format line numbers properly."""
    if session == 0:
        return str(line)
    return "%s#%s" % (session, line)
