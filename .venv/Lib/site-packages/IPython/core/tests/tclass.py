"""Simple script to be run *twice*, to check reference counting bugs.

See test_run for details."""


import sys

# We want to ensure that while objects remain available for immediate access,
# objects from *previous* runs of the same script get collected, to avoid
# accumulating massive amounts of old references.
class C(object):
    def __init__(self,name):
        self.name = name
        self.p = print
        self.flush_stdout = sys.stdout.flush
        
    def __del__(self):
        self.p('tclass.py: deleting object:',self.name)
        self.flush_stdout()

try:
    name = sys.argv[1]
except IndexError:
    pass
else:
    if name.startswith('C'):
        c = C(name)

#print >> sys.stderr, "ARGV:", sys.argv  # dbg

# This next print statement is NOT debugging, we're making the check on a
# completely separate process so we verify by capturing stdout:
print('ARGV 1-:', sys.argv[1:])
sys.stdout.flush()
