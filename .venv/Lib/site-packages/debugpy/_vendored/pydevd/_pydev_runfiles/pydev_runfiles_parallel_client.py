from _pydev_bundle.pydev_imports import xmlrpclib, _queue
Queue = _queue.Queue
import traceback
import sys
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support_from_params
import threading


#=======================================================================================================================
# ParallelNotification
#=======================================================================================================================
class ParallelNotification(object):

    def __init__(self, method, args, kwargs):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def to_tuple(self):
        return self.method, self.args, self.kwargs


#=======================================================================================================================
# KillServer
#=======================================================================================================================
class KillServer(object):
    pass



#=======================================================================================================================
# ServerComm
#=======================================================================================================================
class ServerComm(threading.Thread):



    def __init__(self, job_id, server):
        self.notifications_queue = Queue()
        threading.Thread.__init__(self)
        self.setDaemon(False) #Wait for all the notifications to be passed before exiting!
        assert job_id is not None
        assert port is not None
        self.job_id = job_id

        self.finished = False
        self.server = server


    def run(self):
        while True:
            kill_found = False
            commands = []
            command = self.notifications_queue.get(block=True)
            if isinstance(command, KillServer):
                kill_found = True
            else:
                assert isinstance(command, ParallelNotification)
                commands.append(command.to_tuple())

            try:
                while True:
                    command = self.notifications_queue.get(block=False) #No block to create a batch.
                    if isinstance(command, KillServer):
                        kill_found = True
                    else:
                        assert isinstance(command, ParallelNotification)
                        commands.append(command.to_tuple())
            except:
                pass #That's OK, we're getting it until it becomes empty so that we notify multiple at once.


            if commands:
                try:
                    #Batch notification.
                    self.server.lock.acquire()
                    try:
                        self.server.notifyCommands(self.job_id, commands)
                    finally:
                        self.server.lock.release()
                except:
                    traceback.print_exc()

            if kill_found:
                self.finished = True
                return



#=======================================================================================================================
# ServerFacade
#=======================================================================================================================
class ServerFacade(object):


    def __init__(self, notifications_queue):
        self.notifications_queue = notifications_queue


    def notifyTestsCollected(self, *args, **kwargs):
        pass #This notification won't be passed


    def notifyTestRunFinished(self, *args, **kwargs):
        pass #This notification won't be passed


    def notifyStartTest(self, *args, **kwargs):
        self.notifications_queue.put_nowait(ParallelNotification('notifyStartTest', args, kwargs))


    def notifyTest(self, *args, **kwargs):
        self.notifications_queue.put_nowait(ParallelNotification('notifyTest', args, kwargs))



#=======================================================================================================================
# run_client
#=======================================================================================================================
def run_client(job_id, port, verbosity, coverage_output_file, coverage_include):
    job_id = int(job_id)

    from _pydev_bundle import pydev_localhost
    server = xmlrpclib.Server('http://%s:%s' % (pydev_localhost.get_localhost(), port))
    server.lock = threading.Lock()


    server_comm = ServerComm(job_id, server)
    server_comm.start()

    try:
        server_facade = ServerFacade(server_comm.notifications_queue)
        from _pydev_runfiles import pydev_runfiles
        from _pydev_runfiles import pydev_runfiles_xml_rpc
        pydev_runfiles_xml_rpc.set_server(server_facade)

        #Starts None and when the 1st test is gotten, it's started (because a server may be initiated and terminated
        #before receiving any test -- which would mean a different process got all the tests to run).
        coverage = None

        try:
            tests_to_run = [1]
            while tests_to_run:
                #Investigate: is it dangerous to use the same xmlrpclib server from different threads?
                #It seems it should be, as it creates a new connection for each request...
                server.lock.acquire()
                try:
                    tests_to_run = server.GetTestsToRun(job_id)
                finally:
                    server.lock.release()

                if not tests_to_run:
                    break

                if coverage is None:
                    _coverage_files, coverage = start_coverage_support_from_params(
                        None, coverage_output_file, 1, coverage_include)


                files_to_tests = {}
                for test in tests_to_run:
                    filename_and_test = test.split('|')
                    if len(filename_and_test) == 2:
                        files_to_tests.setdefault(filename_and_test[0], []).append(filename_and_test[1])

                configuration = pydev_runfiles.Configuration(
                    '',
                    verbosity,
                    None,
                    None,
                    None,
                    files_to_tests,
                    1, #Always single job here
                    None,

                    #The coverage is handled in this loop.
                    coverage_output_file=None,
                    coverage_include=None,
                )
                test_runner = pydev_runfiles.PydevTestRunner(configuration)
                sys.stdout.flush()
                test_runner.run_tests(handle_coverage=False)
        finally:
            if coverage is not None:
                coverage.stop()
                coverage.save()


    except:
        traceback.print_exc()
    server_comm.notifications_queue.put_nowait(KillServer())



#=======================================================================================================================
# main
#=======================================================================================================================
if __name__ == '__main__':
    if len(sys.argv) -1 == 3:
        job_id, port, verbosity = sys.argv[1:]
        coverage_output_file, coverage_include = None, None

    elif len(sys.argv) -1 == 5:
        job_id, port, verbosity, coverage_output_file, coverage_include = sys.argv[1:]

    else:
        raise AssertionError('Could not find out how to handle the parameters: '+sys.argv[1:])

    job_id = int(job_id)
    port = int(port)
    verbosity = int(verbosity)
    run_client(job_id, port, verbosity, coverage_output_file, coverage_include)


