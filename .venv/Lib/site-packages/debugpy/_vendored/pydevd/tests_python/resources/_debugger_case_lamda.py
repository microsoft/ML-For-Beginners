call = lambda b:b  # Break here will hit once at creation time and then at each call.
call(1)
call(2)

print('TEST SUCEEDED')