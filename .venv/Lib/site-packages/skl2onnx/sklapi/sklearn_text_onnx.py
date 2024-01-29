# SPDX-License-Identifier: Apache-2.0

from .. import update_registered_converter
from ..shape_calculators.text_vectorizer import (
    calculate_sklearn_text_vectorizer_output_shapes,
)
from ..operator_converters.text_vectoriser import convert_sklearn_text_vectorizer
from ..operator_converters.tfidf_vectoriser import convert_sklearn_tfidf_vectoriser
from .sklearn_text import TraceableCountVectorizer, TraceableTfidfVectorizer


def register():
    """Register converter for TraceableCountVectorizer,
    TraceableTfidfVectorizer."""
    update_registered_converter(
        TraceableCountVectorizer,
        "Skl2onnxTraceableCountVectorizer",
        calculate_sklearn_text_vectorizer_output_shapes,
        convert_sklearn_text_vectorizer,
        options={
            "tokenexp": None,
            "separators": None,
            "nan": [True, False],
            "keep_empty_string": [True, False],
            "locale": None,
        },
    )

    update_registered_converter(
        TraceableTfidfVectorizer,
        "Skl2onnxTraceableTfidfVectorizer",
        calculate_sklearn_text_vectorizer_output_shapes,
        convert_sklearn_tfidf_vectoriser,
        options={
            "tokenexp": None,
            "separators": None,
            "nan": [True, False],
            "keep_empty_string": [True, False],
            "locale": None,
        },
    )
