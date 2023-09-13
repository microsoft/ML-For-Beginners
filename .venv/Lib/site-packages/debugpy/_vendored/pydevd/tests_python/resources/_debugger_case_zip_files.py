from zipped import zipped_contents
try:
    code = zipped_contents.call_in_zip.__code__
except AttributeError:
    code = zipped_contents.call_in_zip.func_code
assert 'myzip.zip' in code.co_filename
zipped_contents.call_in_zip()

from zipped2 import zipped_contents2
try:
    code = zipped_contents2.call_in_zip2.__code__
except AttributeError:
    code = zipped_contents2.call_in_zip2.func_code
assert 'myzip2.egg!' in code.co_filename
zipped_contents2.call_in_zip2()

print('TEST SUCEEDED!')
