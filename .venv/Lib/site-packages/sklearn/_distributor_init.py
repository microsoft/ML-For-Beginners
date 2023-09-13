
'''Helper to preload vcomp140.dll and msvcp140.dll to prevent
"not found" errors.

Once vcomp140.dll and msvcp140.dll are
preloaded, the namespace is made available to any subsequent
vcomp140.dll and msvcp140.dll. This is
created as part of the scripts that build the wheel.
'''


import os
import os.path as op
from ctypes import WinDLL


if os.name == "nt":
    libs_path = op.join(op.dirname(__file__), ".libs")
    vcomp140_dll_filename = op.join(libs_path, "vcomp140.dll")
    msvcp140_dll_filename = op.join(libs_path, "msvcp140.dll")
    WinDLL(op.abspath(vcomp140_dll_filename))
    WinDLL(op.abspath(msvcp140_dll_filename))
