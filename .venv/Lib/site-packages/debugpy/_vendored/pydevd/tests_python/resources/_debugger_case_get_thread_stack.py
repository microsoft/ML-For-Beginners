import threading
event_set = False
inner_started = False

def method():
    global inner_started
    inner_started = True
    while not event_set:
        import time
        time.sleep(.1)
        
t = threading.Thread(target=method)
t.start()
while not inner_started:
    import time
    time.sleep(.1)

print('break here')
event_set = True
t.join()
print('TEST SUCEEDED!')