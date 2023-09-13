def Call():
    b = True
    while b:  # expected
        # requested
        pass  # Note: until 3.10 a pass didn't generate a line event, but starting at 3.10, it does...
        break


if __name__ == '__main__':
    Call()
    print('TEST SUCEEDED!')
