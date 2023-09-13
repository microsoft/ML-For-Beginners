# -*- coding: utf-8 -*-
"""Tools for coloring text in ANSI terminals.
"""

#*****************************************************************************
#       Copyright (C) 2002-2006 Fernando Perez. <fperez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#*****************************************************************************

__all__ = ['TermColors','InputTermColors','ColorScheme','ColorSchemeTable']

import os

from IPython.utils.ipstruct import Struct

color_templates = (
        # Dark colors
        ("Black"       , "0;30"),
        ("Red"         , "0;31"),
        ("Green"       , "0;32"),
        ("Brown"       , "0;33"),
        ("Blue"        , "0;34"),
        ("Purple"      , "0;35"),
        ("Cyan"        , "0;36"),
        ("LightGray"   , "0;37"),
        # Light colors
        ("DarkGray"    , "1;30"),
        ("LightRed"    , "1;31"),
        ("LightGreen"  , "1;32"),
        ("Yellow"      , "1;33"),
        ("LightBlue"   , "1;34"),
        ("LightPurple" , "1;35"),
        ("LightCyan"   , "1;36"),
        ("White"       , "1;37"),
        # Blinking colors.  Probably should not be used in anything serious.
        ("BlinkBlack"  , "5;30"),
        ("BlinkRed"    , "5;31"),
        ("BlinkGreen"  , "5;32"),
        ("BlinkYellow" , "5;33"),
        ("BlinkBlue"   , "5;34"),
        ("BlinkPurple" , "5;35"),
        ("BlinkCyan"   , "5;36"),
        ("BlinkLightGray", "5;37"),
        )

def make_color_table(in_class):
    """Build a set of color attributes in a class.

    Helper function for building the :class:`TermColors` and
    :class`InputTermColors`.
    """
    for name,value in color_templates:
        setattr(in_class,name,in_class._base % value)

class TermColors:
    """Color escape sequences.

    This class defines the escape sequences for all the standard (ANSI?)
    colors in terminals. Also defines a NoColor escape which is just the null
    string, suitable for defining 'dummy' color schemes in terminals which get
    confused by color escapes.

    This class should be used as a mixin for building color schemes."""

    NoColor = ''  # for color schemes in color-less terminals.
    Normal = '\033[0m'   # Reset normal coloring
    _base  = '\033[%sm'  # Template for all other colors

# Build the actual color table as a set of class attributes:
make_color_table(TermColors)

class InputTermColors:
    """Color escape sequences for input prompts.

    This class is similar to TermColors, but the escapes are wrapped in \\001
    and \\002 so that readline can properly know the length of each line and
    can wrap lines accordingly.  Use this class for any colored text which
    needs to be used in input prompts, such as in calls to raw_input().

    This class defines the escape sequences for all the standard (ANSI?)
    colors in terminals. Also defines a NoColor escape which is just the null
    string, suitable for defining 'dummy' color schemes in terminals which get
    confused by color escapes.

    This class should be used as a mixin for building color schemes."""

    NoColor = ''  # for color schemes in color-less terminals.

    if os.name == 'nt' and os.environ.get('TERM','dumb') == 'emacs':
        # (X)emacs on W32 gets confused with \001 and \002 so we remove them
        Normal = '\033[0m'   # Reset normal coloring
        _base  = '\033[%sm'  # Template for all other colors
    else:
        Normal = '\001\033[0m\002'   # Reset normal coloring
        _base  = '\001\033[%sm\002'  # Template for all other colors

# Build the actual color table as a set of class attributes:
make_color_table(InputTermColors)

class NoColors:
    """This defines all the same names as the colour classes, but maps them to
    empty strings, so it can easily be substituted to turn off colours."""
    NoColor = ''
    Normal  = ''

for name, value in color_templates:
    setattr(NoColors, name, '')

class ColorScheme:
    """Generic color scheme class. Just a name and a Struct."""
    def __init__(self,__scheme_name_,colordict=None,**colormap):
        self.name = __scheme_name_
        if colordict is None:
            self.colors = Struct(**colormap)
        else:
            self.colors = Struct(colordict)

    def copy(self,name=None):
        """Return a full copy of the object, optionally renaming it."""
        if name is None:
            name = self.name
        return ColorScheme(name, self.colors.dict())

class ColorSchemeTable(dict):
    """General class to handle tables of color schemes.

    It's basically a dict of color schemes with a couple of shorthand
    attributes and some convenient methods.

    active_scheme_name -> obvious
    active_colors -> actual color table of the active scheme"""

    def __init__(self, scheme_list=None, default_scheme=''):
        """Create a table of color schemes.

        The table can be created empty and manually filled or it can be
        created with a list of valid color schemes AND the specification for
        the default active scheme.
        """

        # create object attributes to be set later
        self.active_scheme_name = ''
        self.active_colors = None

        if scheme_list:
            if default_scheme == '':
                raise ValueError('you must specify the default color scheme')
            for scheme in scheme_list:
                self.add_scheme(scheme)
            self.set_active_scheme(default_scheme)

    def copy(self):
        """Return full copy of object"""
        return ColorSchemeTable(self.values(),self.active_scheme_name)

    def add_scheme(self,new_scheme):
        """Add a new color scheme to the table."""
        if not isinstance(new_scheme,ColorScheme):
            raise ValueError('ColorSchemeTable only accepts ColorScheme instances')
        self[new_scheme.name] = new_scheme

    def set_active_scheme(self,scheme,case_sensitive=0):
        """Set the currently active scheme.

        Names are by default compared in a case-insensitive way, but this can
        be changed by setting the parameter case_sensitive to true."""

        scheme_names = list(self.keys())
        if case_sensitive:
            valid_schemes = scheme_names
            scheme_test = scheme
        else:
            valid_schemes = [s.lower() for s in scheme_names]
            scheme_test = scheme.lower()
        try:
            scheme_idx = valid_schemes.index(scheme_test)
        except ValueError as e:
            raise ValueError('Unrecognized color scheme: ' + scheme + \
                  '\nValid schemes: '+str(scheme_names).replace("'', ",'')) from e
        else:
            active = scheme_names[scheme_idx]
            self.active_scheme_name = active
            self.active_colors = self[active].colors
            # Now allow using '' as an index for the current active scheme
            self[''] = self[active]
