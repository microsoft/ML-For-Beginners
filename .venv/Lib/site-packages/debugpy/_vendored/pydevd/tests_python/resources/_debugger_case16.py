# this test requires numpy to be installed
import numpy

def main():
    smallarray = numpy.arange(100) * 1 + 1j
    bigarray = numpy.arange(100000).reshape((10,10000)) # 100 thousand
    hugearray = numpy.arange(10000000)  # 10 million

    pass  # location of breakpoint after all arrays defined

main()
print('TEST SUCEEDED')
