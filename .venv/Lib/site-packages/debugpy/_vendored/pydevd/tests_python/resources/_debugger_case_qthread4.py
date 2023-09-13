try:
    from PySide import QtCore
except:
    try:
        from PySide2 import QtCore
    except:
        try:
            from PyQt4 import QtCore
        except:
            from PyQt5 import QtCore


class TestObject(QtCore.QObject):
    """
    Test class providing some non-argument signal
    """

    try:
        testSignal = QtCore.Signal()  # @UndefinedVariable
    except:
        testSignal = QtCore.pyqtSignal()  # @UndefinedVariable


class TestThread(QtCore.QThread):

    def run(self):
        QtCore.QThread.sleep(4)
        print('Done sleeping')


def on_start():
    print('On start called1')
    print('On start called2')


app = QtCore.QCoreApplication([])
some_thread = TestThread()
some_object = TestObject()

# connect QThread.started to the signal
some_thread.started.connect(some_object.testSignal)
some_object.testSignal.connect(on_start)
some_thread.finished.connect(app.quit)

some_thread.start()
app.exec_()
print('TEST SUCEEDED!')
