if False:
    config = None
    
    
# See: https://docs.google.com/document/d/1CCSaRiIWCLgbD3OwmuKsRoHHDfBffbROWyVWWL0ZXN4/edit
if ':' not in config.version_id:
    # The default server version_id does not contain ':'
    import json 
    import os
    import sys
    
    startup = config.python_config.startup_args
    if not startup:
        raise AssertionError('Expected --python_startup_args to be passed from the pydev debugger.')
    
    setup = json.loads(startup) 
    pydevd_path = setup['pydevd']
    sys.path.append(os.path.dirname(pydevd_path))
     
    import pydevd
    pydevd.settrace(setup['client'], port=setup['port'], suspend=False, trace_only_current_thread=False)
