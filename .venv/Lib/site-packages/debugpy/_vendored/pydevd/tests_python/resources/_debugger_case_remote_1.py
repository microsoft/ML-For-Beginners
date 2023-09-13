if __name__ == '__main__':
    import subprocess
    import sys
    import os
    import _debugger_case_remote_2
    args = sys.argv[1:]
    port = int(args.pop(0))

    access_token = None
    client_access_token = None
    while args:
        if args[0] == '--access-token':
            access_token = args[1]
            args = args[2:]
        elif args[0] == '--client-access-token':
            client_access_token = args[1]
            args = args[2:]
        else:
            raise AssertionError('Unable to handle args: %s' % (sys.argv[1:]))

    root_dirname = os.path.dirname(os.path.dirname(__file__))

    if root_dirname not in sys.path:
        sys.path.append(root_dirname)

    import pydevd

    print('before pydevd.settrace')
    sys.stdout.flush()
    pydevd.settrace(
        port=port,
        patch_multiprocessing=True,
        access_token=access_token,
        client_access_token=client_access_token,
    )
    print('after pydevd.settrace')
    sys.stdout.flush()
    f = _debugger_case_remote_2.__file__
    if f.endswith('.pyc'):
        f = f[:-1]
    elif f.endswith('$py.class'):
        f = f[:-len('$py.class')] + '.py'
    print('before call')
    sys.stdout.flush()
    subprocess.check_call([sys.executable, '-u', f])
    print('after call')
    sys.stdout.flush()
