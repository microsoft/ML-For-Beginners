# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import locale as pylocale
import unicodedata
import warnings

import numpy as np

from onnx.reference.op_run import OpRun, RuntimeTypeError


class StringNormalizer(OpRun):
    """
    The operator is not really threadsafe as python cannot
    play with two locales at the same time. stop words
    should not be implemented here as the tokenization
    usually happens after this steps.
    """

    def _run(  # type: ignore
        self,
        x,
        case_change_action=None,
        is_case_sensitive=None,
        locale=None,
        stopwords=None,
    ):
        slocale = locale
        if stopwords is None:
            raw_stops = set()
            stops = set()
        else:
            raw_stops = set(stopwords)
            if case_change_action == "LOWER":
                stops = {w.lower() for w in stopwords}
            elif case_change_action == "UPPER":
                stops = {w.upper() for w in stopwords}
            else:
                stops = set(stopwords)
        res = np.empty(x.shape, dtype=x.dtype)
        if len(x.shape) == 2:
            for i in range(0, x.shape[1]):
                self._run_column(
                    x[:, i],
                    res[:, i],
                    slocale=slocale,
                    stops=stops,
                    raw_stops=raw_stops,
                    is_case_sensitive=is_case_sensitive,
                    case_change_action=case_change_action,
                )
        elif len(x.shape) == 1:
            self._run_column(
                x,
                res,
                slocale=slocale,
                stops=stops,
                raw_stops=raw_stops,
                is_case_sensitive=is_case_sensitive,
                case_change_action=case_change_action,
            )
        else:
            raise RuntimeTypeError("x must be a matrix or a vector.")
        if len(res.shape) == 2 and res.shape[0] == 1:
            res = np.array([[w for w in res.tolist()[0] if len(w) > 0]])
            if res.shape[1] == 0:
                res = np.array([[""]])
        elif len(res.shape) == 1:
            res = np.array([w for w in res.tolist() if len(w) > 0])
            if len(res) == 0:
                res = np.array([""])
        return (res,)

    @staticmethod
    def _run_column(  # type: ignore
        cin,
        cout,
        slocale=None,
        stops=None,
        raw_stops=None,
        is_case_sensitive=None,
        case_change_action=None,
    ):
        if pylocale.getlocale() != slocale:
            try:
                pylocale.setlocale(pylocale.LC_ALL, slocale)
            except pylocale.Error as e:
                warnings.warn(
                    f"Unknown local setting {slocale!r} (current: {pylocale.getlocale()!r}) - {e!r}.",
                    stacklevel=1,
                )
        cout[:] = cin[:]

        for i in range(0, cin.shape[0]):
            if isinstance(cout[i], float):
                # nan
                cout[i] = ""
            else:
                cout[i] = StringNormalizer.strip_accents_unicode(cout[i])

        if is_case_sensitive and len(stops) > 0:
            for i in range(0, cin.shape[0]):
                cout[i] = StringNormalizer._remove_stopwords(cout[i], raw_stops)

        if case_change_action == "LOWER":
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].lower()
        elif case_change_action == "UPPER":
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].upper()
        elif case_change_action != "NONE":
            raise RuntimeError(
                f"Unknown option for case_change_action: {case_change_action!r}."
            )

        if not is_case_sensitive and len(stops) > 0:
            for i in range(0, cin.shape[0]):
                cout[i] = StringNormalizer._remove_stopwords(cout[i], stops)

        return cout

    @staticmethod
    def _remove_stopwords(text, stops):  # type: ignore
        spl = text.split(" ")
        return " ".join(filter(lambda s: s not in stops, spl))

    @staticmethod
    def strip_accents_unicode(s):  # type: ignore
        """
        Transforms accentuated unicode symbols into their simple counterpart.
        Source: `sklearn/feature_extraction/text.py
        <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
        feature_extraction/text.py#L115>`_.

        :param s: string
            The string to strip
        :return: the cleaned string
        """
        try:
            # If `s` is ASCII-compatible, then it does not contain any accented
            # characters and we can avoid an expensive list comprehension
            s.encode("ASCII", errors="strict")
            return s
        except UnicodeEncodeError:
            normalized = unicodedata.normalize("NFKD", s)
            s = "".join([c for c in normalized if not unicodedata.combining(c)])
            return s
