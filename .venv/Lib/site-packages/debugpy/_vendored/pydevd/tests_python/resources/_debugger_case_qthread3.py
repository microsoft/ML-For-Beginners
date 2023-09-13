import time
import sys

try:
    try:
        from PySide import QtCore  # @UnresolvedImport
    except:
        from PySide2 import QtCore  # @UnresolvedImport
except:
    try:
        from PyQt4 import QtCore
    except:
        from PyQt5 import QtCore

# Using a QRunnable
# http://doc.qt.nokia.com/latest/qthreadpool.html
# Note that a QRunnable isn't a subclass of QObject and therefore does
# not provide signals and slots.
class Runnable(QtCore.QRunnable):

    def run(self):
        count = 0
        app = QtCore.QCoreApplication.instance()
        while count < 5:
            print("Increasing")  # break here
            count += 1
        app.quit()


app = QtCore.QCoreApplication([])
runnable = Runnable()
QtCore.QThreadPool.globalInstance().start(runnable)
app.exec_()
QtCore.QThreadPool.globalInstance().waitForDone()
print('TEST SUCEEDED!')
