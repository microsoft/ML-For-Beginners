import unittest

class Test(unittest.TestCase):

    def testProcessCommandLine(self):
        from _pydevd_bundle.pydevd_command_line_handling import process_command_line, setup_to_argv
        setup = process_command_line(['pydevd.py', '--port', '1', '--save-threading'])
        assert setup['save-threading']
        assert setup['port'] == 1
        assert not setup['qt-support']

        argv = setup_to_argv(setup)
        assert argv[0].endswith('pydevd.py') or argv[0].endswith('pydevd$py.class'), 'Expected: %s to end with pydevd.py' % (argv[0],)
        argv = argv[1:]
        assert argv == ['--port', '1', '--save-threading']

    def testProcessCommandLine2(self):
        from _pydevd_bundle.pydevd_command_line_handling import process_command_line, setup_to_argv
        setup = process_command_line(['pydevd.py', '--port', '1', '--qt-support=auto'])
        assert setup['qt-support'] == 'auto'

        setup = process_command_line(['pydevd.py', '--port', '1', '--qt-support'])
        assert setup['qt-support'] == 'auto'

        setup = process_command_line(['pydevd.py', '--port', '1', '--qt-support=pyqt4'])
        assert setup['qt-support'] == 'pyqt4'

        self.assertRaises(ValueError, process_command_line, ['pydevd.py', '--port', '1', '--qt-support=wrong'])
