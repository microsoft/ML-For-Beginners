# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

import onnx
from onnx import NodeProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class TfIdfVectorizerHelper:
    def __init__(self, **params: Any) -> None:
        # Attr names
        mode = "mode"
        min_gram_length = "min_gram_length"
        max_gram_length = "max_gram_length"
        max_skip_count = "max_skip_count"
        ngram_counts = "ngram_counts"
        ngram_indexes = "ngram_indexes"
        pool_int64s = "pool_int64s"

        required_attr = [
            mode,
            min_gram_length,
            max_gram_length,
            max_skip_count,
            ngram_counts,
            ngram_indexes,
            pool_int64s,
        ]

        for i in required_attr:
            assert i in params, f"Missing attribute: {i}"

        self.mode = params[mode]
        self.min_gram_length = params[min_gram_length]
        self.max_gram_length = params[max_gram_length]
        self.max_skip_count = params[max_skip_count]
        self.ngram_counts = params[ngram_counts]
        self.ngram_indexes = params[ngram_indexes]
        self.pool_int64s = params[pool_int64s]

    def make_node_noweights(self) -> NodeProto:
        return onnx.helper.make_node(
            "TfIdfVectorizer",
            inputs=["X"],
            outputs=["Y"],
            mode=self.mode,
            min_gram_length=self.min_gram_length,
            max_gram_length=self.max_gram_length,
            max_skip_count=self.max_skip_count,
            ngram_counts=self.ngram_counts,
            ngram_indexes=self.ngram_indexes,
            pool_int64s=self.pool_int64s,
        )


class TfIdfVectorizer(Base):
    @staticmethod
    def export_tf_only_bigrams_skip0() -> None:
        input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
        output = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_only_bigrams_skip0",
        )

    @staticmethod
    def export_tf_batch_onlybigrams_skip0() -> None:
        input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
        output = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]]
        ).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_batch_onlybigrams_skip0",
        )

    @staticmethod
    def export_tf_onlybigrams_levelempty() -> None:
        input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
        output = np.array([1.0, 1.0, 1.0]).astype(np.float32)

        ngram_counts = np.array([0, 0]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2]).astype(np.int64)
        pool_int64s = np.array([5, 6, 7, 8, 6, 7]).astype(  # unigrams none
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=0,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_onlybigrams_levelempty",
        )

    @staticmethod
    def export_tf_onlybigrams_skip5() -> None:
        input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
        output = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0]).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_onlybigrams_skip5",
        )

    @staticmethod
    def export_tf_batch_onlybigrams_skip5() -> None:
        input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
        output = np.array(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
        ).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=2,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_batch_onlybigrams_skip5",
        )

    @staticmethod
    def export_tf_uniandbigrams_skip5() -> None:
        input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
        output = np.array([0.0, 3.0, 1.0, 0.0, 1.0, 3.0, 1.0]).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=1,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_uniandbigrams_skip5",
        )

    @staticmethod
    def export_tf_batch_uniandbigrams_skip5() -> None:
        input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
        output = np.array(
            [[0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]]
        ).astype(np.float32)

        ngram_counts = np.array([0, 4]).astype(np.int64)
        ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
        pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(  # unigrams
            np.int64
        )  # bigrams

        helper = TfIdfVectorizerHelper(
            mode="TF",
            min_gram_length=1,
            max_gram_length=2,
            max_skip_count=5,
            ngram_counts=ngram_counts,
            ngram_indexes=ngram_indexes,
            pool_int64s=pool_int64s,
        )
        node = helper.make_node_noweights()
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_tfidfvectorizer_tf_batch_uniandbigrams_skip5",
        )
