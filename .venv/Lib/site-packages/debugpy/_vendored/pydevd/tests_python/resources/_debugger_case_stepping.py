def step_over():
    print("don't step here")

def step_into():
    print('step into')

def step_out():
    print('step out')  # Break here 1
    print("don't step here")

def Call():
    step_over()  # Break here 2
    step_into()
    step_out()
    print('stepped out')

if __name__ == '__main__':
    Call()
    print('TEST SUCEEDED!')