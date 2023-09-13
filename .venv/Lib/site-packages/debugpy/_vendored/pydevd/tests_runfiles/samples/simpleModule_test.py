import unittest

def setUpModule():
    raise ValueError("This is an INTENTIONAL value error in setUpModule.")

class SetUpModuleTest(unittest.TestCase):
    
    def setUp(cls):
        pass

    def test_blank(self):
        pass


if __name__ == '__main__':
    unittest.main()
