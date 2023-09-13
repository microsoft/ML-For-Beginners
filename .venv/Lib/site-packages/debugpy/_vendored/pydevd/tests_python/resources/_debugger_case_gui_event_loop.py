class LoopHolder:

    @staticmethod
    def gui_loop():
        print('gui_loop() called')


def call_method():
    from _pydevd_bundle.pydevd_constants import get_global_debugger
    py_db = get_global_debugger()

    # Check state prior to breaking
    assert not py_db.gui_in_use
    assert py_db._installed_gui_support
    assert py_db._gui_event_loop == '__main__.LoopHolder.gui_loop'

    print('break here')

    assert py_db.gui_in_use
    assert py_db._installed_gui_support
    assert py_db._gui_event_loop == '__main__.LoopHolder.gui_loop'


if __name__ == '__main__':
    call_method()
    print('TEST SUCEEDED!')
