import sys

def versionok_for_gui():
    ''' Return True if running Python is suitable for GUI Event Integration and deeper IPython integration '''
    # We require Python 2.6+ ...
    if sys.hexversion < 0x02060000:
        return False
    # Or Python 3.2+
    if sys.hexversion >= 0x03000000 and sys.hexversion < 0x03020000:
        return False
    # Not supported under Jython nor IronPython
    if sys.platform.startswith("java") or sys.platform.startswith('cli'):
        return False

    return True

