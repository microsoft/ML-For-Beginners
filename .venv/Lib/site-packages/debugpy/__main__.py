# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import sys


if __name__ == "__main__":
    # debugpy can also be invoked directly rather than via -m. In this case, the first
    # entry on sys.path is the one added automatically by Python for the directory
    # containing this file. This means that import debugpy will not work, since we need
    # the parent directory of debugpy/ to be in sys.path, rather than debugpy/ itself.
    #
    # The other issue is that many other absolute imports will break, because they
    # will be resolved relative to debugpy/ - e.g. `import debugger` will then try
    # to import debugpy/debugger.py.
    #
    # To fix both, we need to replace the automatically added entry such that it points
    # at parent directory of debugpy/ instead of debugpy/ itself, import debugpy with that
    # in sys.path, and then remove the first entry entry altogether, so that it doesn't
    # affect any further imports we might do. For example, suppose the user did:
    #
    #   python /foo/bar/debugpy ...
    #
    # At the beginning of this script, sys.path will contain "/foo/bar/debugpy" as the
    # first entry. What we want is to replace it with "/foo/bar', then import debugpy
    # with that in effect, and then remove the replaced entry before any more
    # code runs. The imported debugpy module will remain in sys.modules, and thus all
    # future imports of it or its submodules will resolve accordingly.
    if "debugpy" not in sys.modules:
        # Do not use dirname() to walk up - this can be a relative path, e.g. ".".
        sys.path[0] = sys.path[0] + "/../"
        import debugpy  # noqa

        del sys.path[0]

    from debugpy.server import cli

    cli.main()
