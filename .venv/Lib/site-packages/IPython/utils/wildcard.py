# -*- coding: utf-8 -*-
"""Support for wildcard pattern matching in object inspection.

Authors
-------
- Jörgen Stenarson <jorgen.stenarson@bostream.nu>
- Thomas Kluyver
"""

#*****************************************************************************
#       Copyright (C) 2005 Jörgen Stenarson <jorgen.stenarson@bostream.nu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#*****************************************************************************

import re
import types

from IPython.utils.dir2 import dir2

def create_typestr2type_dicts(dont_include_in_type2typestr=["lambda"]):
    """Return dictionaries mapping lower case typename (e.g. 'tuple') to type
    objects from the types package, and vice versa."""
    typenamelist = [tname for tname in dir(types) if tname.endswith("Type")]
    typestr2type, type2typestr = {}, {}

    for tname in typenamelist:
        name = tname[:-4].lower()          # Cut 'Type' off the end of the name
        obj = getattr(types, tname)
        typestr2type[name] = obj
        if name not in dont_include_in_type2typestr:
            type2typestr[obj] = name
    return typestr2type, type2typestr

typestr2type, type2typestr = create_typestr2type_dicts()

def is_type(obj, typestr_or_type):
    """is_type(obj, typestr_or_type) verifies if obj is of a certain type. It
    can take strings or actual python types for the second argument, i.e.
    'tuple'<->TupleType. 'all' matches all types.

    TODO: Should be extended for choosing more than one type."""
    if typestr_or_type == "all":
        return True
    if type(typestr_or_type) == type:
        test_type = typestr_or_type
    else:
        test_type = typestr2type.get(typestr_or_type, False)
    if test_type:
        return isinstance(obj, test_type)
    return False

def show_hidden(str, show_all=False):
    """Return true for strings starting with single _ if show_all is true."""
    return show_all or str.startswith("__") or not str.startswith("_")

def dict_dir(obj):
    """Produce a dictionary of an object's attributes. Builds on dir2 by
    checking that a getattr() call actually succeeds."""
    ns = {}
    for key in dir2(obj):
       # This seemingly unnecessary try/except is actually needed
       # because there is code out there with metaclasses that
       # create 'write only' attributes, where a getattr() call
       # will fail even if the attribute appears listed in the
       # object's dictionary.  Properties can actually do the same
       # thing.  In particular, Traits use this pattern
       try:
           ns[key] = getattr(obj, key)
       except AttributeError:
           pass
    return ns

def filter_ns(ns, name_pattern="*", type_pattern="all", ignore_case=True,
            show_all=True):
    """Filter a namespace dictionary by name pattern and item type."""
    pattern = name_pattern.replace("*",".*").replace("?",".")
    if ignore_case:
        reg = re.compile(pattern+"$", re.I)
    else:
        reg = re.compile(pattern+"$")

    # Check each one matches regex; shouldn't be hidden; of correct type.
    return dict((key,obj) for key, obj in ns.items() if reg.match(key) \
                                            and show_hidden(key, show_all) \
                                            and is_type(obj, type_pattern) )

def list_namespace(namespace, type_pattern, filter, ignore_case=False, show_all=False):
    """Return dictionary of all objects in a namespace dictionary that match
    type_pattern and filter."""
    pattern_list=filter.split(".")
    if len(pattern_list) == 1:
       return filter_ns(namespace, name_pattern=pattern_list[0],
                        type_pattern=type_pattern,
                        ignore_case=ignore_case, show_all=show_all)
    else:
        # This is where we can change if all objects should be searched or
        # only modules. Just change the type_pattern to module to search only
        # modules
        filtered = filter_ns(namespace, name_pattern=pattern_list[0],
                            type_pattern="all",
                            ignore_case=ignore_case, show_all=show_all)
        results = {}
        for name, obj in filtered.items():
            ns = list_namespace(dict_dir(obj), type_pattern,
                                ".".join(pattern_list[1:]),
                                ignore_case=ignore_case, show_all=show_all)
            for inner_name, inner_obj in ns.items():
                results["%s.%s"%(name,inner_name)] = inner_obj
        return results
