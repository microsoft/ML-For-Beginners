# Based on: https://github.com/ESSS/pytest-regressions (License: MIT)
# Created copy because we need Python 2.6 which is not available on pytest-regressions.
# Note: only used for testing.

# encoding: UTF-8
import difflib
import pytest
import sys
from functools import partial

if sys.version_info[0] <= 2:
    from pathlib2 import Path
else:
    from pathlib import Path

FORCE_REGEN = False


@pytest.fixture
def original_datadir(request):
    # Method from: https://github.com/gabrielcnr/pytest-datadir
    # License: MIT
    import os.path
    return Path(os.path.splitext(request.module.__file__)[0])


@pytest.fixture
def datadir(original_datadir, tmpdir):
    # Method from: https://github.com/gabrielcnr/pytest-datadir
    # License: MIT
    import shutil
    result = Path(str(tmpdir.join(original_datadir.stem)))
    if original_datadir.is_dir():
        shutil.copytree(str(original_datadir), str(result))
    else:
        result.mkdir()
    return result


@pytest.fixture
def data_regression(datadir, original_datadir, request):
    return DataRegressionFixture(datadir, original_datadir, request)


def check_text_files(obtained_fn, expected_fn, fix_callback=lambda x: x, encoding=None):
    """
    Compare two files contents. If the files differ, show the diff and write a nice HTML
    diff file into the data directory.
    :param Path obtained_fn: path to obtained file during current testing.
    :param Path expected_fn: path to the expected file, obtained from previous testing.
    :param str encoding: encoding used to open the files.
    :param callable fix_callback:
        A callback to "fix" the contents of the obtained (first) file.
        This callback receives a list of strings (lines) and must also return a list of lines,
        changed as needed.
        The resulting lines will be used to compare with the contents of expected_fn.
    """
    __tracebackhide__ = True

    obtained_fn = Path(obtained_fn)
    expected_fn = Path(expected_fn)
    obtained_lines = fix_callback(obtained_fn.read_text(encoding=encoding).splitlines())
    expected_lines = expected_fn.read_text(encoding=encoding).splitlines()

    if obtained_lines != expected_lines:
        diff_lines = list(difflib.unified_diff(expected_lines, obtained_lines))
        if len(diff_lines) <= 500:
            html_fn = obtained_fn.with_suffix(".diff.html")
            try:
                differ = difflib.HtmlDiff()
                html_diff = differ.make_file(
                    fromlines=expected_lines,
                    fromdesc=expected_fn,
                    tolines=obtained_lines,
                    todesc=obtained_fn,
                )
            except Exception as e:
                html_fn = "(failed to generate html diff: %s)" % e
            else:
                html_fn.write_text(html_diff, encoding="UTF-8")

            diff = ["FILES DIFFER:", str(expected_fn), str(obtained_fn)]
            diff += ["HTML DIFF: %s" % html_fn]
            diff += diff_lines
            raise AssertionError("\n".join(diff))
        else:
            # difflib has exponential scaling and for thousands of lines it starts to take minutes to render
            # the HTML diff.
            msg = [
                "Files are different, but diff is too big (%s lines)" % (len(diff_lines),),
                "- obtained: %s" % (obtained_fn,),
                "- expected: %s" % (expected_fn,),
            ]
            raise AssertionError("\n".join(msg))


def perform_regression_check(
    datadir,
    original_datadir,
    request,
    check_fn,
    dump_fn,
    extension,
    basename=None,
    fullpath=None,
    obtained_filename=None,
    dump_aux_fn=lambda filename: [],
):
    """
    First run of this check will generate a expected file. Following attempts will always try to
    match obtained files with that expected file.
    :param Path datadir: Fixture embed_data.
    :param Path original_datadir: Fixture embed_data.
    :param SubRequest request: Pytest request object.
    :param callable check_fn: A function that receives as arguments, respectively, absolute path to
        obtained file and absolute path to expected file. It must assert if contents of file match.
        Function can safely assume that obtained file is already dumped and only care about
        comparison.
    :param callable dump_fn: A function that receive an absolute file path as argument. Implementor
        must dump file in this path.
    :param callable dump_aux_fn: A function that receives the same file path as ``dump_fn``, but may
        dump additional files to help diagnose this regression later (for example dumping image of
        3d views and plots to compare later). Must return the list of file names written (used to display).
    :param six.text_type extension: Extension of files compared by this check.
    :param six.text_type obtained_filename: complete path to use to write the obtained file. By
        default will prepend `.obtained` before the file extension.
    ..see: `data_regression.Check` for `basename` and `fullpath` arguments.
    """
    import re

    assert not (basename and fullpath), "pass either basename or fullpath, but not both"

    __tracebackhide__ = True

    if basename is None:
        basename = re.sub(r"[\W]", "_", request.node.name)

    if fullpath:
        filename = source_filename = Path(fullpath)
    else:
        filename = datadir / (basename + extension)
        source_filename = original_datadir / (basename + extension)

    def make_location_message(banner, filename, aux_files):
        msg = [banner, "- %s" % (filename,)]
        if aux_files:
            msg.append("Auxiliary:")
            msg += ["- %s" % (x,) for x in aux_files]
        return "\n".join(msg)

    if not filename.is_file():
        source_filename.parent.mkdir(parents=True, exist_ok=True)
        dump_fn(source_filename)
        aux_created = dump_aux_fn(source_filename)

        msg = make_location_message(
            "File not found in data directory, created:", source_filename, aux_created
        )
        pytest.fail(msg)
    else:
        if obtained_filename is None:
            if fullpath:
                obtained_filename = (datadir / basename).with_suffix(
                    ".obtained" + extension
                )
            else:
                obtained_filename = filename.with_suffix(".obtained" + extension)

        dump_fn(obtained_filename)

        try:
            check_fn(obtained_filename, filename)
        except AssertionError:
            if FORCE_REGEN:
                dump_fn(source_filename)
                aux_created = dump_aux_fn(source_filename)
                msg = make_location_message(
                    "Files differ and FORCE_REGEN set, regenerating file at:",
                    source_filename,
                    aux_created,
                )
                pytest.fail(msg)
            else:
                dump_aux_fn(obtained_filename)
                raise


class DataRegressionFixture(object):
    """
    Implementation of `data_regression` fixture.
    """

    def __init__(self, datadir, original_datadir, request):
        """
        :type datadir: Path
        :type original_datadir: Path
        :type request: FixtureRequest
        """
        self.request = request
        self.datadir = datadir
        self.original_datadir = original_datadir

    def check(self, data_dict, basename=None, fullpath=None):
        """
        Checks the given dict against a previously recorded version, or generate a new file.
        :param dict data_dict: any yaml serializable dict.
        :param str basename: basename of the file to test/record. If not given the name
            of the test is used.
            Use either `basename` or `fullpath`.
        :param str fullpath: complete path to use as a reference file. This option
            will ignore ``datadir`` fixture when reading *expected* files but will still use it to
            write *obtained* files. Useful if a reference file is located in the session data dir for example.
        ``basename`` and ``fullpath`` are exclusive.
        """
        __tracebackhide__ = True

        def dump(filename):
            """Dump dict contents to the given filename"""
            import json

            s = json.dumps(data_dict, sort_keys=True, indent=4)
            if isinstance(s, bytes):
                s = s.decode('utf-8')

            s = u'\n'.join([line.rstrip() for line in s.splitlines()])
            s = s.encode('utf-8')

            with filename.open("wb") as f:
                f.write(s)

        perform_regression_check(
            datadir=self.datadir,
            original_datadir=self.original_datadir,
            request=self.request,
            check_fn=partial(check_text_files, encoding="UTF-8"),
            dump_fn=dump,
            extension=".json",
            basename=basename,
            fullpath=fullpath,
        )
