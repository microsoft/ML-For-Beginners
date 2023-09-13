def Call2():
    print('Call2')

def Call1(a):
    print('Call1')
    
if __name__ == '__main__':
    Call1(Call2())
    print('TEST SUCEEDED!')
