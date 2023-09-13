from tests_python.resource_path_translation import other


def call_this():
    print('break here')


def main():
    other.call_me_back1(call_this)
    print('TEST SUCEEDED!')


if __name__ == '__main__':
    main()
