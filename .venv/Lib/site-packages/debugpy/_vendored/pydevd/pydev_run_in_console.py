'''
Entry point module to run a file in the interactive console.
'''
import os
import sys
import traceback
from pydevconsole import InterpreterInterface, process_exec_queue, start_console_server, init_mpl_in_console
from _pydev_bundle._pydev_saved_modules import threading, _queue

from _pydev_bundle import pydev_imports
from _pydevd_bundle.pydevd_utils import save_main_module
from _pydev_bundle.pydev_console_utils import StdIn
from pydevd_file_utils import get_fullname


def run_file(file, globals=None, locals=None, is_module=False):
    module_name = None
    entry_point_fn = None
    if is_module:
        file, _, entry_point_fn = file.partition(':')
        module_name = file
        filename = get_fullname(file)
        if filename is None:
            sys.stderr.write("No module named %s\n" % file)
            return
        else:
            file = filename

    if os.path.isdir(file):
        new_target = os.path.join(file, '__main__.py')
        if os.path.isfile(new_target):
            file = new_target

    if globals is None:
        m = save_main_module(file, 'pydev_run_in_console')

        globals = m.__dict__
        try:
            globals['__builtins__'] = __builtins__
        except NameError:
            pass  # Not there on Jython...

    if locals is None:
        locals = globals

    if not is_module:
        sys.path.insert(0, os.path.split(file)[0])

    print('Running %s' % file)
    try:
        if not is_module:
            pydev_imports.execfile(file, globals, locals)  # execute the script
        else:
            # treat ':' as a seperator between module and entry point function
            # if there is no entry point we run we same as with -m switch. Otherwise we perform
            # an import and execute the entry point
            if entry_point_fn:
                mod = __import__(module_name, level=0, fromlist=[entry_point_fn], globals=globals, locals=locals)
                func = getattr(mod, entry_point_fn)
                func()
            else:
                # Run with the -m switch
                from _pydevd_bundle import pydevd_runpy
                pydevd_runpy._run_module_as_main(module_name)
    except:
        traceback.print_exc()

    return globals


def skip_successful_exit(*args):
    """ System exit in file shouldn't kill interpreter (i.e. in `timeit`)"""
    if len(args) == 1 and args[0] in (0, None):
        pass
    else:
        raise SystemExit(*args)


def process_args(argv):
    setup_args = {'file': '', 'module': False}

    setup_args['port'] = argv[1]
    del argv[1]
    setup_args['client_port'] = argv[1]
    del argv[1]

    module_flag = "--module"
    if module_flag in argv:
        i = argv.index(module_flag)
        if i != -1:
            setup_args['module'] = True
            setup_args['file'] = argv[i + 1]
            del sys.argv[i]
    else:
        setup_args['file'] = argv[1]

    del argv[0]

    return setup_args


#=======================================================================================================================
# main
#=======================================================================================================================
if __name__ == '__main__':
    setup = process_args(sys.argv)

    port = setup['port']
    client_port = setup['client_port']
    file = setup['file']
    is_module = setup['module']

    from _pydev_bundle import pydev_localhost

    if int(port) == 0 and int(client_port) == 0:
        (h, p) = pydev_localhost.get_socket_name()
        client_port = p

    host = pydev_localhost.get_localhost()

    # replace exit (see comments on method)
    # note that this does not work in jython!!! (sys method can't be replaced).
    sys.exit = skip_successful_exit

    connect_status_queue = _queue.Queue()
    interpreter = InterpreterInterface(host, int(client_port), threading.current_thread(), connect_status_queue=connect_status_queue)

    server_thread = threading.Thread(target=start_console_server,
                                     name='ServerThread',
                                     args=(host, int(port), interpreter))
    server_thread.daemon = True
    server_thread.start()

    sys.stdin = StdIn(interpreter, host, client_port, sys.stdin)

    init_mpl_in_console(interpreter)

    try:
        success = connect_status_queue.get(True, 60)
        if not success:
            raise ValueError()
    except:
        sys.stderr.write("Console server didn't start\n")
        sys.stderr.flush()
        sys.exit(1)

    globals = run_file(file, None, None, is_module)

    interpreter.get_namespace().update(globals)

    interpreter.ShowConsole()

    process_exec_queue(interpreter)
