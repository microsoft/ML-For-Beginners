# SPDX-License-Identifier: Apache-2.0


import warnings
from collections import OrderedDict, Counter
import numpy as np
from ..common._apply_operation import apply_cast, apply_reshape, apply_identity
from ..common._registration import register_converter
from ..common._topology import Scope, Operator
from ..common._container import ModelComponentContainer
from ..common.data_types import guess_proto_type, StringTensorType
from ..proto import onnx_proto
from ..algebra.onnx_ops import OnnxStringNormalizer


def _intelligent_split(text, op, tokenizer, existing):
    """
    Splits text into tokens. *scikit-learn*
    merges tokens with ``' '.join(tokens)``
    to name ngrams. ``'a  b'`` could be ``('a ', 'b')``
    or ``('a', ' b')``.
    See `ngram sequence
    <https://github.com/scikit-learn/scikit-learn/blob/master/
    sklearn/feature_extraction/text.py#L169>`_.
    """
    if op.analyzer == "word":
        if op.ngram_range[0] == op.ngram_range[1] == 1:
            spl = [text]
        elif op.ngram_range[0] == 1 and len(text) >= 2:
            # Every element is in the vocabulary.
            # Naive method
            p1 = len(text) - len(text.lstrip())
            p2_ = len(text) - len(text.rstrip())
            if p2_ == 0:
                p2 = len(text)
            else:
                p2 = -p2_
            spl = text[p1:p2].split()
            if len(spl) <= 1:
                spl = [text]
            else:
                spl[0] = " " * p1 + spl[0]
                spl[-1] = spl[-1] + " " * p2_
            exc = None
            if len(spl) == 1:
                pass
            elif len(spl) == 2:
                if spl[0] not in op.vocabulary_ or spl[1] not in op.vocabulary_:
                    # This is neceassarily a single token.
                    spl = [text]
                elif spl[0] in op.vocabulary_ and spl[1] in op.vocabulary_:
                    # ambiguity
                    # w1, w2 can be either a 2-grams, either a token.
                    # Usually, ' ' is not part of any token.
                    pass
            elif len(spl) == 3:
                stok = (all([s in op.vocabulary_ for s in spl]), spl)
                spl12 = (
                    spl[2] in op.vocabulary_
                    and (spl[0] + " " + spl[1]) in op.vocabulary_,
                    [spl[0] + " " + spl[1], spl[2]],
                )
                spl23 = (
                    spl[0] in op.vocabulary_
                    and (spl[1] + " " + spl[2]) in op.vocabulary_,
                    [spl[0], spl[1] + " " + spl[2]],
                )
                c = Counter(map(lambda t: t[0], [stok, spl12, spl23]))
                if c.get(True, -1) == 0:
                    spl = [text]
                found = [el[1] for el in [stok, spl12, spl23] if el[0]]
                if len(found) == 1:
                    spl = found[0]
                elif len(found) == 0:
                    spl = [text]
                elif stok[0]:
                    # By default, we assume the token is just the sum of
                    # single words.
                    pass
                else:
                    exc = (
                        "More than one decomposition in tokens: ["
                        + ", ".join(map(lambda t: "-".join(t), found))
                        + "]."
                    )
            elif any(map(lambda g: g in op.vocabulary_, spl)):
                # TODO: handle this case with an algorithm
                # which is able to break a string into
                # known substrings.
                exc = "Unable to identify tokens in n-grams."
            if exc:
                raise RuntimeError(
                    "Unable to split n-grams '{}' into tokens. "
                    "{} This happens when a token contain "
                    "spaces. Token '{}' may be a token or a n-gram '{}'."
                    "".format(text, exc, text, spl)
                )
        else:
            # We reuse the tokenizer hoping that will clear
            # ambiguities but this might be slow.
            spl = tokenizer(text)
    else:
        spl = list(text)

    spl = tuple(spl)
    if spl in existing:
        raise RuntimeError(
            f"The converter cannot guess how to split expression "
            f"{text!r} into tokens. This case happens when tokens have "
            f"spaces."
        )
    if op.ngram_range[0] == 1 and (len(op.ngram_range) == 1 or op.ngram_range[1] > 1):
        # All grams should be existing in the vocabulary.
        for g in spl:
            if g not in op.vocabulary_:
                raise RuntimeError(
                    "Unable to split n-grams '{}' into tokens {} "
                    "existing in the vocabulary. Token '{}' does not "
                    "exist in the vocabulary."
                    ".".format(text, spl, g)
                )
    existing.add(spl)
    return spl


def convert_sklearn_text_vectorizer(
    scope: Scope, operator: Operator, container: ModelComponentContainer
):
    """
    Converters for class
    `TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/
    sklearn.feature_extraction.text.TfidfVectorizer.html>`_.
    The current implementation is a work in progress and the ONNX version
    does not produce the exact same results. The converter lets the user
    change some of its parameters.

    Additional options
    ------------------

    tokenexp: string
        The default will change to true in version 1.6.0.
        The tokenizer splits into words using this regular
        expression or the regular expression specified by
        *scikit-learn* is the value is an empty string.
        See also note below.
        Default value: None
    separators: list of separators
        These separators are used to split a string into words.
        Options *separators* is ignore if options *tokenexp* is not None.
        Default value: ``[' ', '[.]', '\\\\?', ',', ';', ':', '\\\\!']``.
    locale:
        The locale is not mentioned in scikit-object. This option can be
        used to change the value for parameter `locale` of ONNX operator
        `StringNormalizer`.

    Example (from :ref:`l-example-tfidfvectorizer`):

    ::

        seps = {TfidfVectorizer: {"separators": [' ', '[.]', '\\\\?', ',', ';',
                                                 ':', '!', '\\\\(', '\\\\)',
                                                 '\\n', '\\\\"', "'", "-",
                                                 "\\\\[", "\\\\]", "@"]}}
        model_onnx = convert_sklearn(pipeline, "tfidf",
                                     initial_types=[("input", StringTensorType([None, 2]))],
                                     options=seps)

    The default regular expression of the tokenizer is ``(?u)\\\\b\\\\w\\\\w+\\\\b``
    (see `re <https://docs.python.org/3/library/re.html>`_).
    This expression may not supported by the library handling the backend.
    `onnxruntime <https://github.com/Microsoft/onnxruntime>`_ uses
    `re2 <https://github.com/google/re2>`_. You may need to switch
    to a custom tokenizer based on
    `python wrapper for re2 <https://pypi.org/project/re2/>`_
    or its sources `pyre2 <https://github.com/facebook/pyre2>`_
    (`syntax <https://github.com/google/re2/blob/master/doc/syntax.txt>`_).
    If the regular expression is not specified and if
    the instance of TfidfVectorizer is using the default
    pattern ``(?u)\\\\b\\\\w\\\\w+\\\\b``, it is replaced by
    ``[a-zA-Z0-9_]+``. Any other case has to be
    manually handled.

    Regular expression ``[^\\\\\\\\n]`` is used to split
    a sentance into character (and not works) if ``analyser=='char'``.
    The mode ``analyser=='char_wb'`` is not implemented.

    .. versionchanged:: 1.6
        Parameters have been renamed: *sep* into *separators*,
        *regex* into *tokenexp*.
    ````

    """  # noqa
    op = operator.raw_operator

    if container.target_opset is not None and container.target_opset < 9:
        raise RuntimeError(
            "Converter for '{}' only works for opset >= 9."
            "".format(op.__class__.__name__)
        )

    if op.analyzer == "char_wb":
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only tokenizer='word' is fully supported. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    if op.analyzer == "char":
        warnings.warn(
            "The conversion of CountVectorizer may not work. "
            "only tokenizer='word' is fully supported. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.",
            UserWarning,
        )
    if op.strip_accents is not None:
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only strip_accents=None is supported. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )

    options = container.get_options(
        op,
        dict(
            separators="DEFAULT",
            tokenexp=None,
            nan=False,
            keep_empty_string=False,
            locale=None,
        ),
    )
    if set(options) != {"separators", "tokenexp", "nan", "keep_empty_string", "locale"}:
        raise RuntimeError(
            "Unknown option {} for {}".format(set(options) - {"separators"}, type(op))
        )

    if op.analyzer == "word":
        default_pattern = "(?u)\\b\\w\\w+\\b"
        if options["separators"] == "DEFAULT" and options["tokenexp"] is None:
            regex = op.token_pattern
            if regex == default_pattern:
                regex = "[a-zA-Z0-9_]+"
            default_separators = None
        elif options["tokenexp"] is not None:
            if options["tokenexp"]:
                regex = options["tokenexp"]
            else:
                regex = op.token_pattern
                if regex == default_pattern:
                    regex = "[a-zA-Z0-9_]+"
            default_separators = None
        else:
            regex = None
            default_separators = options["separators"]
    else:
        if options["separators"] != "DEFAULT":
            raise RuntimeError(
                "Option separators has no effect " "if analyser != 'word'."
            )
        regex = options["tokenexp"] if options["tokenexp"] else "."
        default_separators = None

    if op.preprocessor is not None:
        raise NotImplementedError(
            "Custom preprocessor cannot be converted into ONNX. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    if op.tokenizer is not None:
        raise NotImplementedError(
            "Custom tokenizer cannot be converted into ONNX. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )
    if op.strip_accents is not None:
        raise NotImplementedError(
            "Operator StringNormalizer cannot remove accents. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues."
        )

    if hasattr(op, "stop_words_"):
        stop_words = op.stop_words_ | (set(op.stop_words) if op.stop_words else set())
    else:
        stop_words = set()
    for w in stop_words:
        if not isinstance(w, str):
            raise TypeError(
                f"One stop word is not a string {w!r} in stop_words={stop_words}."
            )

    if op.lowercase or stop_words:
        if len(operator.input_full_names) != 1:
            raise RuntimeError(
                "Only one input is allowed, found {}.".format(operator.input_full_names)
            )

        # StringNormalizer
        op_type = "StringNormalizer"
        attrs = {"name": scope.get_unique_operator_name(op_type)}
        normalized = scope.get_unique_variable_name("normalized")
        if container.target_opset >= 10:
            attrs.update(
                {
                    "case_change_action": "LOWER",
                    "is_case_sensitive": not op.lowercase,
                }
            )
            op_version = 10
            domain = ""
        else:
            attrs.update(
                {
                    "casechangeaction": "LOWER",
                    "is_case_sensitive": not op.lowercase,
                }
            )
            op_version = 9
            domain = "com.microsoft"
        if options["locale"] is not None:
            attrs["locale"] = options["locale"]
        opvs = 1 if domain == "com.microsoft" else op_version
        if stop_words:
            attrs["stopwords"] = list(sorted(stop_words))

        if options["keep_empty_string"]:
            del attrs["name"]
            op_norm = OnnxStringNormalizer(
                "text_in",
                op_version=container.target_opset,
                output_names=["text_out"],
                **attrs,
            )
            scan_body = op_norm.to_onnx(
                OrderedDict([("text_in", StringTensorType())]),
                outputs=[("text_out", StringTensorType())],
                target_opset=op_version,
            )

            vector = scope.get_unique_variable_name("vector")
            apply_reshape(
                scope,
                operator.input_full_names[0],
                vector,
                container,
                desired_shape=(-1, 1),
            )
            container.add_node(
                "Scan", vector, normalized, body=scan_body.graph, num_scan_inputs=1
            )
        else:
            flatten = scope.get_unique_variable_name("flattened")
            apply_reshape(
                scope,
                operator.input_full_names[0],
                flatten,
                container,
                desired_shape=(-1,),
            )
            container.add_node(
                op_type, flatten, normalized, op_version=opvs, op_domain=domain, **attrs
            )
    else:
        normalized = operator.input_full_names

    # Tokenizer
    padvalue = "#"
    while padvalue in op.vocabulary_:
        padvalue += "#"

    op_type = "Tokenizer"
    attrs = {"name": scope.get_unique_operator_name(op_type)}
    attrs.update(
        {
            "pad_value": padvalue,
            "mark": False,
            "mincharnum": 1,
        }
    )
    if regex is None:
        attrs["separators"] = default_separators
    else:
        attrs["tokenexp"] = regex

    tokenized = scope.get_unique_variable_name("tokenized")
    container.add_node(
        op_type, normalized, tokenized, op_domain="com.microsoft", **attrs
    )

    # Flatten
    # Tokenizer outputs shape {1, C} or {1, 1, C}.
    # Second shape is not allowed by TfIdfVectorizer.
    # We use Flatten which produces {1, C} in both cases.
    flatt_tokenized = scope.get_unique_variable_name("flattened")
    container.add_node(
        "Flatten",
        tokenized,
        flatt_tokenized,
        name=scope.get_unique_operator_name("Flatten"),
    )
    tokenized = flatt_tokenized

    # Ngram - TfIdfVectorizer
    C = max(op.vocabulary_.values()) + 1
    words = [None for i in range(C)]
    weights = [0 for i in range(C)]
    for k, v in op.vocabulary_.items():
        words[v] = k
        weights[v] = 1.0
    mode = "TF"

    # Scikit-learn sorts n-grams by alphabetical order..
    # onnx assumes it is sorted by n.
    tokenizer = op.build_tokenizer()
    split_words = []
    existing = set()
    errors = []
    for w in words:
        if isinstance(w, tuple):
            # TraceableCountVectorizer, TraceableTfIdfVectorizer
            spl = list(w)
            w = " ".join(w)
        else:
            # CountVectorizer, TfIdfVectorizer
            try:
                spl = _intelligent_split(w, op, tokenizer, existing)
            except RuntimeError as e:
                errors.append(e)
                continue
        split_words.append((spl, w))
    if len(errors) > 0:
        err = "\n".join(map(str, errors))
        raise RuntimeError(
            f"There were ambiguities between n-grams and tokens. "
            f"{len(errors)} errors occurred. You can fix it by using "
            f"class Traceable{op.__class__.__name__}.\n"
            f"You can learn more at https://github.com/scikit-learn/"
            f"scikit-learn/issues/13733.\n{err}"
        )

    ng_split_words = sorted([(len(a[0]), a[0], i) for i, a in enumerate(split_words)])
    key_indices = [a[2] for a in ng_split_words]
    ngcounts = [0 for i in range(op.ngram_range[0])]

    words = list(ng_split_words[0][1])
    for i in range(1, len(ng_split_words)):
        if ng_split_words[i - 1][0] != ng_split_words[i][0]:
            ngcounts.append(len(words))
        words.extend(ng_split_words[i][1])

    weights_ = [weights[a[2]] for a in ng_split_words]
    weights = list(weights_)
    for i, ind in enumerate(key_indices):
        weights[ind] = weights_[i]

    # Create the node.
    attrs = {"name": scope.get_unique_operator_name("TfIdfVectorizer")}
    attrs.update(
        {
            "min_gram_length": op.ngram_range[0],
            "max_gram_length": op.ngram_range[1],
            "mode": mode,
            "max_skip_count": 0,
            "pool_strings": words,
            "ngram_indexes": key_indices,
            "ngram_counts": ngcounts,
            "weights": list(map(np.float32, weights)),
        }
    )
    output = scope.get_unique_variable_name("output")

    proto_dtype = guess_proto_type(operator.inputs[0].type)
    if proto_dtype != onnx_proto.TensorProto.DOUBLE:
        proto_dtype = onnx_proto.TensorProto.FLOAT

    if proto_dtype == onnx_proto.TensorProto.DOUBLE:
        output_tf = scope.get_unique_variable_name("cast_result")
    else:
        output_tf = output

    if container.target_opset < 9:
        op_type = "Ngram"
        container.add_node(
            op_type, tokenized, output_tf, op_domain="com.microsoft", **attrs
        )
    else:
        op_type = "TfIdfVectorizer"
        container.add_node(
            op_type, tokenized, output_tf, op_domain="", op_version=9, **attrs
        )

    if proto_dtype == onnx_proto.TensorProto.DOUBLE:
        apply_cast(scope, output_tf, output, container, to=proto_dtype)

    if op.binary:
        cast_result_name = scope.get_unique_variable_name("cast_result")
        output_name = scope.get_unique_variable_name("output_name")

        apply_cast(
            scope, output, cast_result_name, container, to=onnx_proto.TensorProto.BOOL
        )
        apply_cast(
            scope,
            cast_result_name,
            output_name,
            container,
            to=onnx_proto.TensorProto.FLOAT,
        )
        output = output_name

    options = container.get_options(op, dict(nan=False))
    replace_by_nan = options.get("nan", False)
    if replace_by_nan:
        # This part replaces all null values by nan.
        cst_nan_name = scope.get_unique_variable_name("nan_name")
        container.add_initializer(cst_nan_name, proto_dtype, [1], [np.nan])
        cst_zero_name = scope.get_unique_variable_name("zero_name")
        container.add_initializer(cst_zero_name, proto_dtype, [1], [0])

        mask_name = scope.get_unique_variable_name("mask_name")
        container.add_node(
            "Equal",
            [output, cst_zero_name],
            mask_name,
            name=scope.get_unique_operator_name("Equal"),
        )

        where_name = scope.get_unique_variable_name("where_name")
        container.add_node(
            "Where",
            [mask_name, cst_nan_name, output],
            where_name,
            name=scope.get_unique_operator_name("Where"),
        )
        output = where_name

    apply_identity(scope, output, operator.output_full_names, container)


register_converter(
    "SklearnCountVectorizer",
    convert_sklearn_text_vectorizer,
    options={
        "tokenexp": None,
        "separators": None,
        "nan": [True, False],
        "keep_empty_string": [True, False],
        "locale": None,
    },
)
