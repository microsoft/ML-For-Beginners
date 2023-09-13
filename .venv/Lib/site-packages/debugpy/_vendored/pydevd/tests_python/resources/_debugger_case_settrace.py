def ask_for_stop(use_back):
    import pydevd
    if use_back:
        pydevd.settrace(stop_at_frame=sys._getframe().f_back)
    else:
        pydevd.settrace()
    print('Will stop here if use_back==False.')


def outer_method():
    ask_for_stop(True)
    print('will stop here.')
    ask_for_stop(False)


if __name__ == '__main__':
    import os
    import sys
    root_dirname = os.path.dirname(os.path.dirname(__file__))
    
    if root_dirname not in sys.path:
        sys.path.append(root_dirname)

    outer_method()        
    print('TEST SUCEEDED!')
    
