import pytest

from nltk import config_megam
from nltk.classify.rte_classify import RTEFeatureExtractor, rte_classifier, rte_features
from nltk.corpus import rte as rte_corpus

expected_from_rte_feature_extration = """
alwayson        => True
ne_hyp_extra    => 0
ne_overlap      => 1
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 3
word_overlap    => 3

alwayson        => True
ne_hyp_extra    => 0
ne_overlap      => 1
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 2
word_overlap    => 1

alwayson        => True
ne_hyp_extra    => 1
ne_overlap      => 1
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 1
word_overlap    => 2

alwayson        => True
ne_hyp_extra    => 1
ne_overlap      => 0
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 6
word_overlap    => 2

alwayson        => True
ne_hyp_extra    => 1
ne_overlap      => 0
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 4
word_overlap    => 0

alwayson        => True
ne_hyp_extra    => 1
ne_overlap      => 0
neg_hyp         => 0
neg_txt         => 0
word_hyp_extra  => 3
word_overlap    => 1
"""


class TestRTEClassifier:
    # Test the feature extraction method.
    def test_rte_feature_extraction(self):
        pairs = rte_corpus.pairs(["rte1_dev.xml"])[:6]
        test_output = [
            f"{key:<15} => {rte_features(pair)[key]}"
            for pair in pairs
            for key in sorted(rte_features(pair))
        ]
        expected_output = expected_from_rte_feature_extration.strip().split("\n")
        # Remove null strings.
        expected_output = list(filter(None, expected_output))
        assert test_output == expected_output

    # Test the RTEFeatureExtractor object.
    def test_feature_extractor_object(self):
        rtepair = rte_corpus.pairs(["rte3_dev.xml"])[33]
        extractor = RTEFeatureExtractor(rtepair)

        assert extractor.hyp_words == {"member", "China", "SCO."}
        assert extractor.overlap("word") == set()
        assert extractor.overlap("ne") == {"China"}
        assert extractor.hyp_extra("word") == {"member"}

    # Test the RTE classifier training.
    def test_rte_classification_without_megam(self):
        # Use a sample size for unit testing, since we
        # don't need to fully train these classifiers
        clf = rte_classifier("IIS", sample_N=100)
        clf = rte_classifier("GIS", sample_N=100)

    def test_rte_classification_with_megam(self):
        try:
            config_megam()
        except (LookupError, AttributeError) as e:
            pytest.skip("Skipping tests with dependencies on MEGAM")
        clf = rte_classifier("megam", sample_N=100)
