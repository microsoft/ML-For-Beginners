This folder contains vendored dependencies of the debugger.

Right now this means the 'bytecode' library (MIT license).

To update the version remove the bytecode* contents from this folder and then use:

pip install bytecode --target .

or from master (if needed for some early bugfix):

python -m pip install https://github.com/MatthieuDartiailh/bytecode/archive/main.zip --target .

Then run 'pydevd_fix_code.py' to fix the imports on the vendored file, run its tests (to see
if things are still ok) and commit.

Then, to finish, apply the patch to add the offset to the instructions (bcb8a28669e9178f96f5d71af7259e0674acc47c)

Note: commit the egg-info as a note of the license (force if needed).