if __name__ == '__main__':
    for i in range(2):
        print("one")
        try:
            raise AttributeError() # Breakpoint here
        except:
            pass
    print('TEST SUCEEDED!')