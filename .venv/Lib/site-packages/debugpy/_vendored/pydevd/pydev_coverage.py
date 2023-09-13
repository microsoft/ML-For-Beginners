'''
Entry point module to run code-coverage.
'''


def is_valid_py_file(path):
    '''
    Checks whether the file can be read by the coverage module. This is especially
    needed for .pyx files and .py files with syntax errors.
    '''
    import os

    is_valid = False
    if os.path.isfile(path) and not os.path.splitext(path)[1] == '.pyx':
        try:
            with open(path, 'rb') as f:
                compile(f.read(), path, 'exec')
                is_valid = True
        except:
            pass
    return is_valid


def execute():
    import os
    import sys

    files = None
    if 'combine' not in sys.argv:

        if '--pydev-analyze' in sys.argv:

            # Ok, what we want here is having the files passed through stdin (because
            # there may be too many files for passing in the command line -- we could
            # just pass a dir and make the find files here, but as that's already
            # given in the java side, let's just gather that info here).
            sys.argv.remove('--pydev-analyze')
            s = input()
            s = s.replace('\r', '')
            s = s.replace('\n', '')

            files = []
            invalid_files = []
            for v in s.split('|'):
                if is_valid_py_file(v):
                    files.append(v)
                else:
                    invalid_files.append(v)
            if invalid_files:
                sys.stderr.write('Invalid files not passed to coverage: %s\n'
                                 % ', '.join(invalid_files))

            # Note that in this case we'll already be in the working dir with the coverage files,
            # so, the coverage file location is not passed.

        else:
            # For all commands, the coverage file is configured in pydev, and passed as the first
            # argument in the command line, so, let's make sure this gets to the coverage module.
            os.environ['COVERAGE_FILE'] = sys.argv[1]
            del sys.argv[1]

    try:
        import coverage  # @UnresolvedImport
    except:
        sys.stderr.write('Error: coverage module could not be imported\n')
        sys.stderr.write('Please make sure that the coverage module '
                         '(http://nedbatchelder.com/code/coverage/)\n')
        sys.stderr.write('is properly installed in your interpreter: %s\n' % (sys.executable,))

        import traceback;traceback.print_exc()
        return

    if hasattr(coverage, '__version__'):
        version = tuple(map(int, coverage.__version__.split('.')[:2]))
        if version < (4, 3):
            sys.stderr.write('Error: minimum supported coverage version is 4.3.'
                             '\nFound: %s\nLocation: %s\n'
                             % ('.'.join(str(x) for x in version), coverage.__file__))
            sys.exit(1)
    else:
        sys.stderr.write('Warning: Could not determine version of python module coverage.'
                         '\nEnsure coverage version is >= 4.3\n')

    from coverage.cmdline import main  # @UnresolvedImport

    if files is not None:
        sys.argv.append('xml')
        sys.argv += files

    main()


if __name__ == '__main__':
    execute()
