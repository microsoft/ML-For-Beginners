import sys
import weakref

def set_up():
    observable = Observable()
    observer = Observer()
    observable.add_observer(observer)
    return observable


class Observable(object):
    def __init__(self):
        self.observers = []
        
    def add_observer(self, observer):
        sys.stdout.write( 'observer %s\n' % (observer,))
        ref = weakref.ref(observer)
        self.observers.append(ref)
        sys.stdout.write('weakref: %s\n' % (ref(),))
        
    def Notify(self):
        for o in self.observers:
            o = o()
            
            
            try:
                import gc
            except ImportError:
                o = None #some jython does not have gc, so, there's no sense testing this in it
            else:
                try:
                    gc.get_referrers(o)
                except:
                    o = None #jython and ironpython do not have get_referrers
            
            if o is not None:
                sys.stdout.write('still observing %s\n' % (o,))
                sys.stdout.write('number of referrers: %s\n' % len(gc.get_referrers(o)))
                frame = gc.get_referrers(o)[0]
                frame_referrers = gc.get_referrers(frame)
                sys.stdout.write('frame referrer %s\n' % (frame_referrers,))
                referrers1 = gc.get_referrers(frame_referrers[1])
                sys.stdout.write('%s\n' % (referrers1,))
                sys.stderr.write('TEST FAILED: The observer should have died, even when running in debug\n')
            else:
                sys.stdout.write('TEST SUCEEDED: observer died\n')
                
            sys.stdout.flush()
            sys.stderr.flush()
                
class Observer(object):
    pass

    
def main():
    observable = set_up()
    observable.Notify()
    
    
if __name__ == '__main__':
    main()
