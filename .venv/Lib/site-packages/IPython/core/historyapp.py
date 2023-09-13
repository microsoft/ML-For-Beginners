# encoding: utf-8
"""
An application for managing IPython history.

To be invoked as the `ipython history` subcommand.
"""

import sqlite3
from pathlib import Path

from traitlets.config.application import Application
from .application import BaseIPythonApplication
from traitlets import Bool, Int, Dict
from ..utils.io import ask_yes_no

trim_hist_help = """Trim the IPython history database to the last 1000 entries.

This actually copies the last 1000 entries to a new database, and then replaces
the old file with the new. Use the `--keep=` argument to specify a number
other than 1000.
"""

clear_hist_help = """Clear the IPython history database, deleting all entries.

Because this is a destructive operation, IPython will prompt the user if they
really want to do this. Passing a `-f` flag will force clearing without a
prompt.

This is an handy alias to `ipython history trim --keep=0`
"""


class HistoryTrim(BaseIPythonApplication):
    description = trim_hist_help
    
    backup = Bool(False,
        help="Keep the old history file as history.sqlite.<N>"
        ).tag(config=True)
    
    keep = Int(1000,
        help="Number of recent lines to keep in the database."
        ).tag(config=True)
    
    flags = Dict(dict(
        backup = ({'HistoryTrim' : {'backup' : True}},
            backup.help
        )
    ))

    aliases=Dict(dict(
        keep = 'HistoryTrim.keep'
    ))
    
    def start(self):
        profile_dir = Path(self.profile_dir.location)
        hist_file = profile_dir / "history.sqlite"
        con = sqlite3.connect(hist_file)

        # Grab the recent history from the current database.
        inputs = list(con.execute('SELECT session, line, source, source_raw FROM '
                                'history ORDER BY session DESC, line DESC LIMIT ?', (self.keep+1,)))
        if len(inputs) <= self.keep:
            print("There are already at most %d entries in the history database." % self.keep)
            print("Not doing anything. Use --keep= argument to keep fewer entries")
            return
        
        print("Trimming history to the most recent %d entries." % self.keep)
        
        inputs.pop() # Remove the extra element we got to check the length.
        inputs.reverse()
        if inputs:
            first_session = inputs[0][0]
            outputs = list(con.execute('SELECT session, line, output FROM '
                                       'output_history WHERE session >= ?', (first_session,)))
            sessions = list(con.execute('SELECT session, start, end, num_cmds, remark FROM '
                                        'sessions WHERE session >= ?', (first_session,)))
        con.close()
        
        # Create the new history database.
        new_hist_file = profile_dir / "history.sqlite.new"
        i = 0
        while new_hist_file.exists():
            # Make sure we don't interfere with an existing file.
            i += 1
            new_hist_file = profile_dir / ("history.sqlite.new" + str(i))
        new_db = sqlite3.connect(new_hist_file)
        new_db.execute("""CREATE TABLE IF NOT EXISTS sessions (session integer
                            primary key autoincrement, start timestamp,
                            end timestamp, num_cmds integer, remark text)""")
        new_db.execute("""CREATE TABLE IF NOT EXISTS history
                        (session integer, line integer, source text, source_raw text,
                        PRIMARY KEY (session, line))""")
        new_db.execute("""CREATE TABLE IF NOT EXISTS output_history
                        (session integer, line integer, output text,
                        PRIMARY KEY (session, line))""")
        new_db.commit()


        if inputs:
            with new_db:
                # Add the recent history into the new database.
                new_db.executemany('insert into sessions values (?,?,?,?,?)', sessions)
                new_db.executemany('insert into history values (?,?,?,?)', inputs)
                new_db.executemany('insert into output_history values (?,?,?)', outputs)
        new_db.close()

        if self.backup:
            i = 1
            backup_hist_file = profile_dir / ("history.sqlite.old.%d" % i)
            while backup_hist_file.exists():
                i += 1
                backup_hist_file = profile_dir / ("history.sqlite.old.%d" % i)
            hist_file.rename(backup_hist_file)
            print("Backed up longer history file to", backup_hist_file)
        else:
            hist_file.unlink()
        
        new_hist_file.rename(hist_file)

class HistoryClear(HistoryTrim):
    description = clear_hist_help
    keep = Int(0,
        help="Number of recent lines to keep in the database.")
    
    force = Bool(False,
        help="Don't prompt user for confirmation"
        ).tag(config=True)
    
    flags = Dict(dict(
        force = ({'HistoryClear' : {'force' : True}},
            force.help),
        f = ({'HistoryTrim' : {'force' : True}},
            force.help
        )
    ))
    aliases = Dict()

    def start(self):
        if self.force or ask_yes_no("Really delete all ipython history? ",
                default="no", interrupt="no"):
            HistoryTrim.start(self)

class HistoryApp(Application):
    name = u'ipython-history'
    description = "Manage the IPython history database."

    subcommands = Dict(dict(
        trim = (HistoryTrim, HistoryTrim.description.splitlines()[0]),
        clear = (HistoryClear, HistoryClear.description.splitlines()[0]),
    ))

    def start(self):
        if self.subapp is None:
            print("No subcommand specified. Must specify one of: %s" % \
                                                    (self.subcommands.keys()))
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()
