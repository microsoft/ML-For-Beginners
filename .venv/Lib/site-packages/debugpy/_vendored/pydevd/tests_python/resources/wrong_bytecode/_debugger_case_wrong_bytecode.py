import helper
import sys
import helper
import helper
import helper
import helper
from helper import helper as sa
from helper.helper.helper import NULL
from helper.helper import NULL

Base = NULL()

def MyFunction():
	pass

class MyClassA(object):
	__tablename__ = 'tableA'
	n = sa.NULL(sa.NULL, primary_key=True)

class MyClassB(object):
	__tablename__ = 'tableB'
	n = sa.NULL(sa.NULL, primary_key=True)

class MyClassC(object):
	def __init__(self):
		pass

class MyClassD(object):
	def __init__(self):
		pass

if __name__ == "__main__":
	N = len(sys.argv)  # break here

	if N >= 2: # step 1
		arg1 = sys.argv[1]  # the debugger gets here even though N=1
		print(arg1)
	else:
		arg1 = 'MyString' # step 2

	if N >= 3: # step 3
		arg2 = int(sys.argv[2])  # the debugger gets here even though N=1
	else:
		arg2 = int(0) # step 4

	print(N)  # still N=1 step 5
	# print(arg1)  # if you print this then it changes the debugger behavior above