import time
def Call():
    loop = True # Break here

    count = 0
    while(loop and (count < 200)):
        time.sleep(0.1)
        count += 1 # Pause here and change loop to False


if __name__ == '__main__':
    Call()
    print('TEST SUCEEDED!')