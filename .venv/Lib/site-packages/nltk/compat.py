# Natural Language Toolkit: Compatibility
#
# Copyright (C) 2001-2023 NLTK Project
#
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import os
from functools import wraps

# ======= Compatibility for datasets that care about Python versions ========

# The following datasets have a /PY3 subdirectory containing
# a full copy of the data which has been re-encoded or repickled.
DATA_UPDATES = [
    ("chunkers", "maxent_ne_chunker"),
    ("help", "tagsets"),
    ("taggers", "maxent_treebank_pos_tagger"),
    ("tokenizers", "punkt"),
]

_PY3_DATA_UPDATES = [os.path.join(*path_list) for path_list in DATA_UPDATES]


def add_py3_data(path):
    for item in _PY3_DATA_UPDATES:
        if item in str(path) and "/PY3" not in str(path):
            pos = path.index(item) + len(item)
            if path[pos : pos + 4] == ".zip":
                pos += 4
            path = path[:pos] + "/PY3" + path[pos:]
            break
    return path


# for use in adding /PY3 to the second (filename) argument
# of the file pointers in data.py
def py3_data(init_func):
    def _decorator(*args, **kwargs):
        args = (args[0], add_py3_data(args[1])) + args[2:]
        return init_func(*args, **kwargs)

    return wraps(init_func)(_decorator)
