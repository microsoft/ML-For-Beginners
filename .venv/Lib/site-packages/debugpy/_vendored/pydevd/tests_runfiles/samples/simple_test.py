import unittest

class SampleTest(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_xxxxxx1(self):
        self.fail('Fail test 2')
    def test_xxxxxx2(self):
        pass
    def test_xxxxxx3(self):
        pass
    def test_xxxxxx4(self):
        pass
    def test_non_unique_name(self):
        print('non unique name ran')


class AnotherSampleTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1(self):
        pass
    def test_2(self):
        """ im a doc string"""
        pass
    def todo_not_tested(self):
        '''
        Not there by default!
        '''


if __name__ == '__main__':
#    suite = unittest.makeSuite(SampleTest, 'test')
#    runner = unittest.TextTestRunner( verbosity=3 )
#    runner.run(suite)
    unittest.main()
