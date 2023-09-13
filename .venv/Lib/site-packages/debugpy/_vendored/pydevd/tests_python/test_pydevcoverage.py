import os
import re
import sys
import subprocess
import tempfile
import unittest


#=======================================================================================================================
# Test
#=======================================================================================================================
class Test(unittest.TestCase):
    """
    Unittest for pydev_coverage.py.
    TODO:
      - 'combine' in arguments
      - no 'combine' and no 'pydev-analyze' in arguments
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        project_path = os.path.dirname(os.path.dirname(__file__))
        self._resources_path = os.path.join(project_path, "tests_python", "resources")
        self._coverage_file = os.path.join(project_path, "pydev_coverage.py")

    def _do_analyze(self, files):
        invalid_files = []

        p = subprocess.Popen([sys.executable, self._coverage_file, "--pydev-analyze"],
                             stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        __, stderrdata = p.communicate("|".join(files).encode())

        if stderrdata:
            match = re.search("Invalid files not passed to coverage: (.*?)$",
                              stderrdata.decode(), re.M)  # @UndefinedVariable
            if match:
                invalid_files = [f.strip() for f in match.group(1).split(",")]
        return invalid_files

    def test_pydev_analyze_ok(self):
        ref_valid_files = [__file__,
            os.path.join(self._resources_path, "_debugger_case18.py")]
        ref_invalid_files = []

        invalid_files = self._do_analyze(ref_valid_files)

        self.assertEqual(ref_invalid_files, invalid_files)

    def test_pydev_analyse_non_standard_encoding(self):
        ref_valid_files = [os.path.join(self._resources_path,
                                        "_pydev_coverage_cyrillic_encoding_py%i.py"
                                        % sys.version_info[0])]
        ref_invalid_files = []

        invalid_files = self._do_analyze(ref_valid_files + ref_invalid_files)

        self.assertEqual(ref_invalid_files, invalid_files)

    def test_pydev_analyse_invalid_files(self):
        with tempfile.NamedTemporaryFile(suffix=".pyx") as pyx_file:
            ref_valid_files = []
            ref_invalid_files = [os.path.join(self._resources_path,
                                              "_pydev_coverage_syntax_error.py"),
                                 pyx_file.name]

            invalid_files = self._do_analyze(ref_valid_files + ref_invalid_files)

            self.assertEqual(ref_invalid_files, invalid_files)
