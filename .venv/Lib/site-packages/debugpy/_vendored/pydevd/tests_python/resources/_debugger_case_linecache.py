import linecache

unique_filename = "<foo bar>"


def foo():
    print("on foo")  # breakpoint


script = """
def somemethod():
    foo()
somemethod()
[x for x in range(10)]
"""

linecache.cache[unique_filename] = (
    len(script),
    None,
    script.splitlines(True),
    unique_filename,
)

obj = compile(script, unique_filename, "exec")
exec(obj)
print('TEST SUCEEDED')
