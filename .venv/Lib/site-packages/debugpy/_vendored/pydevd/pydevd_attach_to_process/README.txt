This folder contains the utilities to attach a target process to the pydev debugger.

The main module to be called for the attach is:

attach_pydevd.py

it should be called as;

python attach_pydevd.py --port 5678 --pid 1234

Note that the client is responsible for having a remote debugger alive in the given port for the attach to work.


The binaries are now compiled at:
- https://github.com/fabioz/PyDev.Debugger.binaries/actions
(after generation the binaries are copied to this repo)


To copy:
cd /D X:\PyDev.Debugger
"C:\Program Files\7-Zip\7z" e C:\Users\fabio\Downloads\win_binaries.zip -oX:\PyDev.Debugger\pydevd_attach_to_process * -r -y
"C:\Program Files\7-Zip\7z" e C:\Users\fabio\Downloads\linux_binaries.zip -oX:\PyDev.Debugger\pydevd_attach_to_process * -r -y
"C:\Program Files\7-Zip\7z" e C:\Users\fabio\Downloads\mac_binaries.zip -oX:\PyDev.Debugger\pydevd_attach_to_process * -r -y
git add *.exe
git add *.dll
git add *.dylib
git add *.so
