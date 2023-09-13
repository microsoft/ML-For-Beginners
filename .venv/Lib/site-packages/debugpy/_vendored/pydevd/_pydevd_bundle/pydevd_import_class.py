#Note: code gotten from _pydev_imports_tipper.

import sys

def _imp(name, log=None):
    try:
        return __import__(name)
    except:
        if '.' in name:
            sub = name[0:name.rfind('.')]
            
            if log is not None:
                log.add_content('Unable to import', name, 'trying with', sub)
                log.add_exception()
            
            return _imp(sub, log)
        else:
            s = 'Unable to import module: %s - sys.path: %s' % (str(name), sys.path)
            if log is not None:
                log.add_content(s)
                log.add_exception()
            
            raise ImportError(s)
        

IS_IPY = False
if sys.platform == 'cli':
    IS_IPY = True
    _old_imp = _imp
    def _imp(name, log=None):
        #We must add a reference in clr for .Net
        import clr #@UnresolvedImport
        initial_name = name
        while '.' in name:
            try:
                clr.AddReference(name)
                break #If it worked, that's OK.
            except:
                name = name[0:name.rfind('.')]
        else:
            try:
                clr.AddReference(name)
            except:
                pass #That's OK (not dot net module).
        
        return _old_imp(initial_name, log)
    

def import_name(name, log=None):
    mod = _imp(name, log)

    components = name.split('.')

    old_comp = None
    for comp in components[1:]:
        try:
            #this happens in the following case:
            #we have mx.DateTime.mxDateTime.mxDateTime.pyd
            #but after importing it, mx.DateTime.mxDateTime shadows access to mxDateTime.pyd
            mod = getattr(mod, comp)
        except AttributeError:
            if old_comp != comp:
                raise
        
        old_comp = comp
        
    return mod
    
