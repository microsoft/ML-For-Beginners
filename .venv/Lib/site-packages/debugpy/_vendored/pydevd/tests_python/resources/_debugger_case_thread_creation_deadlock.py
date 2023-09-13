from threading import Thread, Event


def create_thread():
    event = Event()
    event_set = [False]

    def run_thread():
        event_set[0] = True
        event.set()
    
    t = Thread(target=run_thread)
    t.start()
    
    try:
        event.wait(5)
        
        # note: not using `assert event.wait(5)` for py2.6 compatibility.
        assert event_set[0]  
        print('TEST SUCEEDED!')
        return 'create_thread:ok'
    except:
        import traceback;traceback.print_exc()

        
a = 10  # Add breakpoint here and evaluate create_thread()
