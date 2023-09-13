import warnings


class MyClass(object):

    def __getattribute__(self, attr):
        warnings.warn(
            "Deprecation Warning",
            DeprecationWarning
        )

        warnings.warn(
            "Future Warning!",
            FutureWarning,
        )
        return 1


obj = MyClass()

if __name__ == '__main__':
    print('TEST SUCEEDED')  # break here
