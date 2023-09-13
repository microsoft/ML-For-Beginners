#-----------------------------------------------------------------------------
#  Copyright (C) 2012-  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

from pathlib import Path

from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.utils.tempdir import TemporaryWorkingDirectory


def test_named_file_in_temporary_directory():
    with NamedFileInTemporaryDirectory('filename') as file:
        name = file.name
        assert not file.closed
        assert Path(name).exists()
        file.write(b'test')
    assert file.closed
    assert not Path(name).exists()

def test_temporary_working_directory():
    with TemporaryWorkingDirectory() as directory:
        directory_path = Path(directory).resolve()
        assert directory_path.exists()
        assert Path.cwd().resolve() == directory_path
    assert not directory_path.exists()
    assert Path.cwd().resolve() != directory_path
