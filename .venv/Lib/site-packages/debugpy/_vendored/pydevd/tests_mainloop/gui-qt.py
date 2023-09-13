#!/usr/bin/env python
"""Simple Qt4 example to manually test event loop integration.

To run this:
1) Enable the PyDev GUI event loop integration for qt
2) do an execfile on this script
3) ensure you have a working GUI simultaneously with an
   interactive console

Ref: Modified from http://zetcode.com/tutorials/pyqt4/firstprograms/
"""

if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui, QtCore
    
    class SimpleWindow(QtGui.QWidget):
        def __init__(self, parent=None):
            QtGui.QWidget.__init__(self, parent)
    
            self.setGeometry(300, 300, 200, 80)
            self.setWindowTitle('Hello World')
    
            quit = QtGui.QPushButton('Close', self)
            quit.setGeometry(10, 10, 60, 35)
    
            self.connect(quit, QtCore.SIGNAL('clicked()'),
                         self, QtCore.SLOT('close()'))
    
    if __name__ == '__main__':
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtGui.QApplication([])
    
        sw = SimpleWindow()
        sw.show()
