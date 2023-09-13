import unittest

class SetUpClassTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raise ValueError("This is an INTENTIONAL value error in setUpClass.")

    def test_blank(self):
        pass


if __name__ == '__main__':
    unittest.main()
