import sys


def main():
    import subprocess
    process = subprocess.Popen(
        'git status --porcelain'.split(), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    if output:
        if sys.version_info[0] > 2:
            output = output.decode('utf-8')

        files = set()
        for line in output.splitlines():
            filename = line[3:]
            files.add(filename.strip())

        files.discard('.travis_install_python_deps.sh')
        files.discard('miniconda.sh')
        if files:
            # If there are modifications, show a diff of the modifications and fail the script.
            # (we're mostly interested in modifications to the .c generated files by cython).
            print('Found modifications in git:\n%s ' % (output,))
            print('Files: %s' % (files,))
            print('----------- diff -------------')
            subprocess.call('git diff'.split())
            print('----------- end diff -------------')
            sys.exit(1)


if __name__ == '__main__':
    main()
