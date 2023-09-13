# -*- coding: utf-8 -*-
"""Release data for the IPython project."""

#-----------------------------------------------------------------------------
#  Copyright (c) 2008, IPython Development Team.
#  Copyright (c) 2001, Fernando Perez <fernando.perez@colorado.edu>
#  Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
#  Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

# IPython version information.  An empty _version_extra corresponds to a full
# release.  'dev' as a _version_extra string means this is a development
# version
_version_major = 8
_version_minor = 15
_version_patch = 0
_version_extra = ".dev"
# _version_extra = "rc1"
_version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor, _version_patch]

__version__ = '.'.join(map(str, _ver))
if _version_extra:
    __version__ = __version__  + _version_extra

version = __version__  # backwards compatibility name
version_info = (_version_major, _version_minor, _version_patch, _version_extra)

# Change this when incrementing the kernel protocol version
kernel_protocol_version_info = (5, 0)
kernel_protocol_version = "%i.%i" % kernel_protocol_version_info

license = "BSD-3-Clause"

authors = {'Fernando' : ('Fernando Perez','fperez.net@gmail.com'),
           'Janko'    : ('Janko Hauser','jhauser@zscout.de'),
           'Nathan'   : ('Nathaniel Gray','n8gray@caltech.edu'),
           'Ville'    : ('Ville Vainio','vivainio@gmail.com'),
           'Brian'    : ('Brian E Granger', 'ellisonbg@gmail.com'),
           'Min'      : ('Min Ragan-Kelley', 'benjaminrk@gmail.com'),
           'Thomas'   : ('Thomas A. Kluyver', 'takowl@gmail.com'),
           'Jorgen'   : ('Jorgen Stenarson', 'jorgen.stenarson@bostream.nu'),
           'Matthias' : ('Matthias Bussonnier', 'bussonniermatthias@gmail.com'),
           }

author = 'The IPython Development Team'

author_email = 'ipython-dev@python.org'
