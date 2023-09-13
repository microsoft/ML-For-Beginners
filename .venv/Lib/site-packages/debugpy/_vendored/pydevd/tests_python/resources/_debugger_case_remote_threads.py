import threading
if __name__ == '__main__':
    import os
    import sys
    port = int(sys.argv[1])
    root_dirname = os.path.dirname(os.path.dirname(__file__))

    if root_dirname not in sys.path:
        sys.path.append(root_dirname)

    def method(i):
        import time
        wait = True
        while wait:
            time.sleep(.1)  # break here

    threads = [threading.Thread(target=method, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()

    import pydevd
    assert pydevd.get_global_debugger() is None

    print('before pydevd.settrace')
    pydevd.settrace(port=port)
    print('after pydevd.settrace')

    for t in threads:
        t.join()

    print('TEST SUCEEDED!')
