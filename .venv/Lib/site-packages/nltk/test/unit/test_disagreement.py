import unittest

from nltk.metrics.agreement import AnnotationTask


class TestDisagreement(unittest.TestCase):

    """
    Class containing unit tests for nltk.metrics.agreement.Disagreement.
    """

    def test_easy(self):
        """
        Simple test, based on
        https://github.com/foolswood/krippendorffs_alpha/raw/master/krippendorff.pdf.
        """
        data = [
            ("coder1", "dress1", "YES"),
            ("coder2", "dress1", "NO"),
            ("coder3", "dress1", "NO"),
            ("coder1", "dress2", "YES"),
            ("coder2", "dress2", "NO"),
            ("coder3", "dress3", "NO"),
        ]
        annotation_task = AnnotationTask(data)
        self.assertAlmostEqual(annotation_task.alpha(), -0.3333333)

    def test_easy2(self):
        """
        Same simple test with 1 rating removed.
        Removal of that rating should not matter: K-Apha ignores items with
        only 1 rating.
        """
        data = [
            ("coder1", "dress1", "YES"),
            ("coder2", "dress1", "NO"),
            ("coder3", "dress1", "NO"),
            ("coder1", "dress2", "YES"),
            ("coder2", "dress2", "NO"),
        ]
        annotation_task = AnnotationTask(data)
        self.assertAlmostEqual(annotation_task.alpha(), -0.3333333)

    def test_advanced(self):
        """
        More advanced test, based on
        http://www.agreestat.com/research_papers/onkrippendorffalpha.pdf
        """
        data = [
            ("A", "1", "1"),
            ("B", "1", "1"),
            ("D", "1", "1"),
            ("A", "2", "2"),
            ("B", "2", "2"),
            ("C", "2", "3"),
            ("D", "2", "2"),
            ("A", "3", "3"),
            ("B", "3", "3"),
            ("C", "3", "3"),
            ("D", "3", "3"),
            ("A", "4", "3"),
            ("B", "4", "3"),
            ("C", "4", "3"),
            ("D", "4", "3"),
            ("A", "5", "2"),
            ("B", "5", "2"),
            ("C", "5", "2"),
            ("D", "5", "2"),
            ("A", "6", "1"),
            ("B", "6", "2"),
            ("C", "6", "3"),
            ("D", "6", "4"),
            ("A", "7", "4"),
            ("B", "7", "4"),
            ("C", "7", "4"),
            ("D", "7", "4"),
            ("A", "8", "1"),
            ("B", "8", "1"),
            ("C", "8", "2"),
            ("D", "8", "1"),
            ("A", "9", "2"),
            ("B", "9", "2"),
            ("C", "9", "2"),
            ("D", "9", "2"),
            ("B", "10", "5"),
            ("C", "10", "5"),
            ("D", "10", "5"),
            ("C", "11", "1"),
            ("D", "11", "1"),
            ("C", "12", "3"),
        ]
        annotation_task = AnnotationTask(data)
        self.assertAlmostEqual(annotation_task.alpha(), 0.743421052632)

    def test_advanced2(self):
        """
        Same more advanced example, but with 1 rating removed.
        Again, removal of that 1 rating should not matter.
        """
        data = [
            ("A", "1", "1"),
            ("B", "1", "1"),
            ("D", "1", "1"),
            ("A", "2", "2"),
            ("B", "2", "2"),
            ("C", "2", "3"),
            ("D", "2", "2"),
            ("A", "3", "3"),
            ("B", "3", "3"),
            ("C", "3", "3"),
            ("D", "3", "3"),
            ("A", "4", "3"),
            ("B", "4", "3"),
            ("C", "4", "3"),
            ("D", "4", "3"),
            ("A", "5", "2"),
            ("B", "5", "2"),
            ("C", "5", "2"),
            ("D", "5", "2"),
            ("A", "6", "1"),
            ("B", "6", "2"),
            ("C", "6", "3"),
            ("D", "6", "4"),
            ("A", "7", "4"),
            ("B", "7", "4"),
            ("C", "7", "4"),
            ("D", "7", "4"),
            ("A", "8", "1"),
            ("B", "8", "1"),
            ("C", "8", "2"),
            ("D", "8", "1"),
            ("A", "9", "2"),
            ("B", "9", "2"),
            ("C", "9", "2"),
            ("D", "9", "2"),
            ("B", "10", "5"),
            ("C", "10", "5"),
            ("D", "10", "5"),
            ("C", "11", "1"),
            ("D", "11", "1"),
            ("C", "12", "3"),
        ]
        annotation_task = AnnotationTask(data)
        self.assertAlmostEqual(annotation_task.alpha(), 0.743421052632)
