def foo():
    return 1  # breakpoint


def main():
    import linecache
    import time

    code = 'foo()'

    # Intermediate <string> stack frame will have no source.
    eval(code)

    co_filename = '<something>'
    co = compile(code, co_filename, 'exec')

    # Set up the source in linecache. Python doesn't have a public API for
    # this, so we have to hack around it, similar to what IPython does.
    linecache.cache[co_filename] = (
        len(code),
        time.time(),
        [line + '\n' for line in code.splitlines()],
        co_filename,
    )

    # Intermediate <something> stack frame will have source.
    eval(co)


if __name__ == '__main__':
    main()
    print('TEST SUCEEDED!')
