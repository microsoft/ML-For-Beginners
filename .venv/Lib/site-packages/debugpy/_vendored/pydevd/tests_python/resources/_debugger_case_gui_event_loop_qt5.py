def call_method():

    from _pydevd_bundle.pydevd_constants import get_global_debugger
    py_db = get_global_debugger()

    # Check state prior to breaking
    assert not py_db.gui_in_use
    assert py_db._installed_gui_support
    assert py_db._gui_event_loop == 'qt5'

    import  os
    import PySide2
    from PySide2.QtCore import QTimer

    dirname = os.path.dirname(PySide2.__file__)
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

    from PySide2 import QtWidgets

    app = QtWidgets.QApplication([])

    def on_timeout():
        print('on_timeout() called')

    print_timer = QTimer()
    print_timer.timeout.connect(on_timeout)
    print_timer.setInterval(100)
    print_timer.start()

    def on_break():
        print('break here')
        app.quit()

    break_on_timer = QTimer()
    break_on_timer.timeout.connect(on_break)
    break_on_timer.setSingleShot(True)
    break_on_timer.setInterval(50)
    break_on_timer.start()

    app.exec_()  # Run forever until app.quit()

    assert py_db.gui_in_use
    assert py_db._installed_gui_support
    assert py_db._gui_event_loop == 'qt5'


if __name__ == '__main__':
    call_method()
    print('TEST SUCEEDED!')
