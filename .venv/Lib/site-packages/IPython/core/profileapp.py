# encoding: utf-8
"""
An application for managing IPython profiles.

To be invoked as the `ipython profile` subcommand.

Authors:

* Min RK

"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import os

from traitlets.config.application import Application
from IPython.core.application import (
    BaseIPythonApplication, base_flags
)
from IPython.core.profiledir import ProfileDir
from IPython.utils.importstring import import_item
from IPython.paths import get_ipython_dir, get_ipython_package_dir
from traitlets import Unicode, Bool, Dict, observe

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

create_help = """Create an IPython profile by name

Create an ipython profile directory by its name or
profile directory path. Profile directories contain
configuration, log and security related files and are named
using the convention 'profile_<name>'. By default they are
located in your ipython directory. Once created, you will
can edit the configuration files in the profile
directory to configure IPython. Most users will create a
profile directory by name,
`ipython profile create myprofile`, which will put the directory
in `<ipython_dir>/profile_myprofile`.
"""
list_help = """List available IPython profiles

List all available profiles, by profile location, that can
be found in the current working directly or in the ipython
directory. Profile directories are named using the convention
'profile_<profile>'.
"""
profile_help = """Manage IPython profiles

Profile directories contain
configuration, log and security related files and are named
using the convention 'profile_<name>'. By default they are
located in your ipython directory.  You can create profiles
with `ipython profile create <name>`, or see the profiles you
already have with `ipython profile list`

To get started configuring IPython, simply do:

$> ipython profile create

and IPython will create the default profile in <ipython_dir>/profile_default,
where you can edit ipython_config.py to start configuring IPython.

"""

_list_examples = "ipython profile list  # list all profiles"

_create_examples = """
ipython profile create foo         # create profile foo w/ default config files
ipython profile create foo --reset # restage default config files over current
ipython profile create foo --parallel # also stage parallel config files
"""

_main_examples = """
ipython profile create -h  # show the help string for the create subcommand
ipython profile list -h    # show the help string for the list subcommand

ipython locate profile foo # print the path to the directory for profile 'foo'
"""

#-----------------------------------------------------------------------------
# Profile Application Class (for `ipython profile` subcommand)
#-----------------------------------------------------------------------------


def list_profiles_in(path):
    """list profiles in a given root directory"""
    profiles = []

    # for python 3.6+ rewrite to: with os.scandir(path) as dirlist:
    files = os.scandir(path)
    for f in files:
        if f.is_dir() and f.name.startswith('profile_'):
            profiles.append(f.name.split('_', 1)[-1])
    return profiles


def list_bundled_profiles():
    """list profiles that are bundled with IPython."""
    path = os.path.join(get_ipython_package_dir(), u'core', u'profile')
    profiles = []

    # for python 3.6+ rewrite to: with os.scandir(path) as dirlist:
    files =  os.scandir(path)
    for profile in files:
        if profile.is_dir() and profile.name != "__pycache__":
            profiles.append(profile.name)
    return profiles


class ProfileLocate(BaseIPythonApplication):
    description = """print the path to an IPython profile dir"""
    
    def parse_command_line(self, argv=None):
        super(ProfileLocate, self).parse_command_line(argv)
        if self.extra_args:
            self.profile = self.extra_args[0]
    
    def start(self):
        print(self.profile_dir.location)


class ProfileList(Application):
    name = u'ipython-profile'
    description = list_help
    examples = _list_examples

    aliases = Dict({
        'ipython-dir' : 'ProfileList.ipython_dir',
        'log-level' : 'Application.log_level',
    })
    flags = Dict(dict(
        debug = ({'Application' : {'log_level' : 0}},
            "Set Application.log_level to 0, maximizing log output."
        )
    ))

    ipython_dir = Unicode(get_ipython_dir(),
        help="""
        The name of the IPython directory. This directory is used for logging
        configuration (through profiles), history storage, etc. The default
        is usually $HOME/.ipython. This options can also be specified through
        the environment variable IPYTHONDIR.
        """
    ).tag(config=True)


    def _print_profiles(self, profiles):
        """print list of profiles, indented."""
        for profile in profiles:
            print('    %s' % profile)

    def list_profile_dirs(self):
        profiles = list_bundled_profiles()
        if profiles:
            print()
            print("Available profiles in IPython:")
            self._print_profiles(profiles)
            print()
            print("    The first request for a bundled profile will copy it")
            print("    into your IPython directory (%s)," % self.ipython_dir)
            print("    where you can customize it.")
        
        profiles = list_profiles_in(self.ipython_dir)
        if profiles:
            print()
            print("Available profiles in %s:" % self.ipython_dir)
            self._print_profiles(profiles)
        
        profiles = list_profiles_in(os.getcwd())
        if profiles:
            print()
            print(
                "Profiles from CWD have been removed for security reason, see CVE-2022-21699:"
            )

        print()
        print("To use any of the above profiles, start IPython with:")
        print("    ipython --profile=<name>")
        print()

    def start(self):
        self.list_profile_dirs()


create_flags = {}
create_flags.update(base_flags)
# don't include '--init' flag, which implies running profile create in other apps
create_flags.pop('init')
create_flags['reset'] = ({'ProfileCreate': {'overwrite' : True}},
                        "reset config files in this profile to the defaults.")
create_flags['parallel'] = ({'ProfileCreate': {'parallel' : True}},
                        "Include the config files for parallel "
                        "computing apps (ipengine, ipcontroller, etc.)")


class ProfileCreate(BaseIPythonApplication):
    name = u'ipython-profile'
    description = create_help
    examples = _create_examples
    auto_create = Bool(True)
    def _log_format_default(self):
        return "[%(name)s] %(message)s"

    def _copy_config_files_default(self):
        return True

    parallel = Bool(False,
        help="whether to include parallel computing config files"
    ).tag(config=True)

    @observe('parallel')
    def _parallel_changed(self, change):
        parallel_files = [   'ipcontroller_config.py',
                            'ipengine_config.py',
                            'ipcluster_config.py'
                        ]
        if change['new']:
            for cf in parallel_files:
                self.config_files.append(cf)
        else:
            for cf in parallel_files:
                if cf in self.config_files:
                    self.config_files.remove(cf)

    def parse_command_line(self, argv):
        super(ProfileCreate, self).parse_command_line(argv)
        # accept positional arg as profile name
        if self.extra_args:
            self.profile = self.extra_args[0]

    flags = Dict(create_flags)

    classes = [ProfileDir]
    
    def _import_app(self, app_path):
        """import an app class"""
        app = None
        name = app_path.rsplit('.', 1)[-1]
        try:
            app = import_item(app_path)
        except ImportError:
            self.log.info("Couldn't import %s, config file will be excluded", name)
        except Exception:
            self.log.warning('Unexpected error importing %s', name, exc_info=True)
        return app

    def init_config_files(self):
        super(ProfileCreate, self).init_config_files()
        # use local imports, since these classes may import from here
        from IPython.terminal.ipapp import TerminalIPythonApp
        apps = [TerminalIPythonApp]
        for app_path in (
            'ipykernel.kernelapp.IPKernelApp',
        ):
            app = self._import_app(app_path)
            if app is not None:
                apps.append(app)
        if self.parallel:
            from ipyparallel.apps.ipcontrollerapp import IPControllerApp
            from ipyparallel.apps.ipengineapp import IPEngineApp
            from ipyparallel.apps.ipclusterapp import IPClusterStart
            apps.extend([
                IPControllerApp,
                IPEngineApp,
                IPClusterStart,
            ])
        for App in apps:
            app = App()
            app.config.update(self.config)
            app.log = self.log
            app.overwrite = self.overwrite
            app.copy_config_files=True
            app.ipython_dir=self.ipython_dir
            app.profile_dir=self.profile_dir
            app.init_config_files()

    def stage_default_config_file(self):
        pass


class ProfileApp(Application):
    name = u'ipython profile'
    description = profile_help
    examples = _main_examples

    subcommands = Dict(dict(
        create = (ProfileCreate, ProfileCreate.description.splitlines()[0]),
        list = (ProfileList, ProfileList.description.splitlines()[0]),
        locate = (ProfileLocate, ProfileLocate.description.splitlines()[0]),
    ))

    def start(self):
        if self.subapp is None:
            print("No subcommand specified. Must specify one of: %s"%(self.subcommands.keys()))
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()
