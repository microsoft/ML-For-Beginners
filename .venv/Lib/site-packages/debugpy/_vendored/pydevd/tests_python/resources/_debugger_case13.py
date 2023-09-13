
class TestProperty(object):
    def __init__(self, name = "Default"):
        self._x = None
        self.name = name

    def get_name(self):
        return self.__name


    def set_name(self, value):
        self.__name = value


    def del_name(self):
        del self.__name
    name = property(get_name, set_name, del_name, "name's docstring")

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x

def main():
    """
    """
    testObj = TestProperty()
    testObj.x = 10
    val = testObj.x
    
    testObj.name = "Pydev"
    debugType = testObj.name
    print('TEST SUCEEDED!')
    
if __name__ == '__main__':
    main()