import sys
if '--as-module' in sys.argv:
    from . import bar
else:
    import bar
print('Worked')
