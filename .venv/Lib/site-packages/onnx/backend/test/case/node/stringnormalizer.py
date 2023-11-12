# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class StringNormalizer(Base):
    @staticmethod
    def export_nostopwords_nochangecase() -> None:
        input = np.array(["monday", "tuesday"]).astype(object)
        output = input

        # No stopwords. This is a NOOP
        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            is_case_sensitive=1,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_nostopwords_nochangecase",
        )

    @staticmethod
    def export_monday_casesensintive_nochangecase() -> None:
        input = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        output = np.array(["tuesday", "wednesday", "thursday"]).astype(object)
        stopwords = ["monday"]

        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            is_case_sensitive=1,
            stopwords=stopwords,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_export_monday_casesensintive_nochangecase",
        )

    @staticmethod
    def export_monday_casesensintive_lower() -> None:
        input = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        output = np.array(["tuesday", "wednesday", "thursday"]).astype(object)
        stopwords = ["monday"]

        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="LOWER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_export_monday_casesensintive_lower",
        )

    @staticmethod
    def export_monday_casesensintive_upper() -> None:
        input = np.array(["monday", "tuesday", "wednesday", "thursday"]).astype(object)
        output = np.array(["TUESDAY", "WEDNESDAY", "THURSDAY"]).astype(object)
        stopwords = ["monday"]

        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_export_monday_casesensintive_upper",
        )

    @staticmethod
    def export_monday_empty_output() -> None:
        input = np.array(["monday", "monday"]).astype(object)
        output = np.array([""]).astype(object)
        stopwords = ["monday"]

        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            is_case_sensitive=1,
            stopwords=stopwords,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_export_monday_empty_output",
        )

    @staticmethod
    def export_monday_insensintive_upper_twodim() -> None:
        input = (
            np.array(
                ["Monday", "tuesday", "wednesday", "Monday", "tuesday", "wednesday"]
            )
            .astype(object)
            .reshape([1, 6])
        )

        # It does upper case cecedille, accented E
        # and german umlaut but fails
        # with german eszett
        output = (
            np.array(["TUESDAY", "WEDNESDAY", "TUESDAY", "WEDNESDAY"])
            .astype(object)
            .reshape([1, 4])
        )
        stopwords = ["monday"]

        node = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["x"],
            outputs=["y"],
            case_change_action="UPPER",
            stopwords=stopwords,
        )
        expect(
            node,
            inputs=[input],
            outputs=[output],
            name="test_strnormalizer_export_monday_insensintive_upper_twodim",
        )
