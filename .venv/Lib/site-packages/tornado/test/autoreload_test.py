import os
import shutil
import subprocess
from subprocess import Popen
import sys
from tempfile import mkdtemp
import textwrap
import time
import unittest


class AutoreloadTest(unittest.TestCase):
    def setUp(self):
        # When these tests fail the output sometimes exceeds the default maxDiff.
        self.maxDiff = 1024

        self.path = mkdtemp()

        # Most test apps run themselves twice via autoreload. The first time it manually triggers
        # a reload (could also do this by touching a file but this is faster since filesystem
        # timestamps are not necessarily high resolution). The second time it exits directly
        # so that the autoreload wrapper (if it is used) doesn't catch it.
        #
        # The last line of each such test's "main" program should be
        #     exec(open("run_twice_magic.py").read())
        self.write_files(
            {
                "run_twice_magic.py": """
                    import os
                    import sys

                    import tornado.autoreload

                    sys.stdout.flush()

                    if "TESTAPP_STARTED" not in os.environ:
                        os.environ["TESTAPP_STARTED"] = "1"
                        tornado.autoreload._reload()
                    else:
                        os._exit(0)
                """
            }
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.path)
        except OSError:
            # Windows disallows deleting files that are in use by
            # another process, and even though we've waited for our
            # child process below, it appears that its lock on these
            # files is not guaranteed to be released by this point.
            # Sleep and try again (once).
            time.sleep(1)
            shutil.rmtree(self.path)

    def write_files(self, tree, base_path=None):
        """Write a directory tree to self.path.

        tree is a dictionary mapping file names to contents, or
        sub-dictionaries representing subdirectories.
        """
        if base_path is None:
            base_path = self.path
        for name, contents in tree.items():
            if isinstance(contents, dict):
                os.mkdir(os.path.join(base_path, name))
                self.write_files(contents, os.path.join(base_path, name))
            else:
                with open(os.path.join(base_path, name), "w", encoding="utf-8") as f:
                    f.write(textwrap.dedent(contents))

    def run_subprocess(self, args):
        # Make sure the tornado module under test is available to the test
        # application
        pythonpath = os.getcwd()
        if "PYTHONPATH" in os.environ:
            pythonpath += os.pathsep + os.environ["PYTHONPATH"]

        p = Popen(
            args,
            stdout=subprocess.PIPE,
            env=dict(os.environ, PYTHONPATH=pythonpath),
            cwd=self.path,
            universal_newlines=True,
            encoding="utf-8",
        )

        # This timeout needs to be fairly generous for pypy due to jit
        # warmup costs.
        for i in range(40):
            if p.poll() is not None:
                break
            time.sleep(0.1)
        else:
            p.kill()
            raise Exception("subprocess failed to terminate")

        out = p.communicate()[0]
        self.assertEqual(p.returncode, 0)
        return out

    def test_reload(self):
        main = """\
import sys

# In module mode, the path is set to the parent directory and we can import testapp.
try:
    import testapp
except ImportError:
    print("import testapp failed")
else:
    print("import testapp succeeded")

spec = getattr(sys.modules[__name__], '__spec__', None)
print(f"Starting {__name__=}, __spec__.name={getattr(spec, 'name', None)}")
exec(open("run_twice_magic.py").read())
"""

        # Create temporary test application
        self.write_files(
            {
                "testapp": {
                    "__init__.py": "",
                    "__main__.py": main,
                },
            }
        )

        # The autoreload wrapper should support all the same modes as the python interpreter.
        # The wrapper itself should have no effect on this test so we try all modes with and
        # without it.
        for wrapper in [False, True]:
            with self.subTest(wrapper=wrapper):
                with self.subTest(mode="module"):
                    if wrapper:
                        base_args = [sys.executable, "-m", "tornado.autoreload"]
                    else:
                        base_args = [sys.executable]
                    # In module mode, the path is set to the parent directory and we can import
                    # testapp. Also, the __spec__.name is set to the fully qualified module name.
                    out = self.run_subprocess(base_args + ["-m", "testapp"])
                    self.assertEqual(
                        out,
                        (
                            "import testapp succeeded\n"
                            + "Starting __name__='__main__', __spec__.name=testapp.__main__\n"
                        )
                        * 2,
                    )

                with self.subTest(mode="file"):
                    out = self.run_subprocess(base_args + ["testapp/__main__.py"])
                    # In file mode, we do not expect the path to be set so we can import testapp,
                    # but when the wrapper is used the -m argument to the python interpreter
                    # does this for us.
                    expect_import = (
                        "import testapp succeeded"
                        if wrapper
                        else "import testapp failed"
                    )
                    # In file mode there is no qualified module spec.
                    self.assertEqual(
                        out,
                        f"{expect_import}\nStarting __name__='__main__', __spec__.name=None\n"
                        * 2,
                    )

                with self.subTest(mode="directory"):
                    # Running as a directory finds __main__.py like a module. It does not manipulate
                    # sys.path but it does set a spec with a name of exactly __main__.
                    out = self.run_subprocess(base_args + ["testapp"])
                    expect_import = (
                        "import testapp succeeded"
                        if wrapper
                        else "import testapp failed"
                    )
                    self.assertEqual(
                        out,
                        f"{expect_import}\nStarting __name__='__main__', __spec__.name=__main__\n"
                        * 2,
                    )

    def test_reload_wrapper_preservation(self):
        # This test verifies that when `python -m tornado.autoreload`
        # is used on an application that also has an internal
        # autoreload, the reload wrapper is preserved on restart.
        main = """\
import sys

# This import will fail if path is not set up correctly
import testapp

if 'tornado.autoreload' not in sys.modules:
    raise Exception('started without autoreload wrapper')

print('Starting')
exec(open("run_twice_magic.py").read())
"""

        self.write_files(
            {
                "testapp": {
                    "__init__.py": "",
                    "__main__.py": main,
                },
            }
        )

        out = self.run_subprocess(
            [sys.executable, "-m", "tornado.autoreload", "-m", "testapp"]
        )
        self.assertEqual(out, "Starting\n" * 2)

    def test_reload_wrapper_args(self):
        main = """\
import os
import sys

print(os.path.basename(sys.argv[0]))
print(f'argv={sys.argv[1:]}')
exec(open("run_twice_magic.py").read())
"""
        # Create temporary test application
        self.write_files({"main.py": main})

        # Make sure the tornado module under test is available to the test
        # application
        out = self.run_subprocess(
            [
                sys.executable,
                "-m",
                "tornado.autoreload",
                "main.py",
                "arg1",
                "--arg2",
                "-m",
                "arg3",
            ],
        )

        self.assertEqual(out, "main.py\nargv=['arg1', '--arg2', '-m', 'arg3']\n" * 2)

    def test_reload_wrapper_until_success(self):
        main = """\
import os
import sys

if "TESTAPP_STARTED" in os.environ:
    print("exiting cleanly")
    sys.exit(0)
else:
    print("reloading")
    exec(open("run_twice_magic.py").read())
"""

        # Create temporary test application
        self.write_files({"main.py": main})

        out = self.run_subprocess(
            [sys.executable, "-m", "tornado.autoreload", "--until-success", "main.py"]
        )

        self.assertEqual(out, "reloading\nexiting cleanly\n")
