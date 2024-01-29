# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from enum import IntEnum
from typing import List

import numpy as np

from onnx.reference.op_run import OpRun


class IntMap(dict):  # type: ignore
    def __init__(self):
        dict.__init__(self)
        self.added_keys = []

    def emplace(self, key, value):
        if not isinstance(key, int):
            raise TypeError(f"key must be a NGramPart not {type(key)}.")
        if not isinstance(value, NgramPart):
            raise TypeError(f"value must be a NGramPart not {type(value)}.")
        if key not in self:
            self.added_keys.append(key)
            self[key] = value
        return self[key]

    def __repr__(self):
        vals = {k: repr(v) for k, v in self.items()}
        rows = ["{"]
        for k, v in sorted(vals.items()):
            if "\n" in v:
                vs = v.split("\n")
                for i, line in enumerate(vs):
                    if i == 0:
                        if line == "{":
                            rows.append(f"  {k}={line}")
                        else:
                            rows.append(f"  {k}={line},")
                    elif i == len(vs) - 1:
                        rows.append(f"  {line}")
                    else:
                        rows.append(f"    {line}")
            else:
                rows.append(f"  {k}={v},")
        rows.append("}")
        return "\n".join(rows)

    @property
    def first_key(self):
        if len(self) == 0:
            raise ValueError("IntMap is empty.")
        return self.added_keys[0]


class NgramPart:
    def __init__(self, nid: int):
        self.id_ = nid  # 0 - means no entry, search for a bigger N
        self._leafs_ = None

    def init(self):
        self._leafs_ = IntMap()  # type: ignore

    def __repr__(self):
        if self.empty():
            return f"NgramPart({self.id_})"
        return f"NgramPart({self.id_}, {self.leafs_!r})"

    def empty(self):
        return self._leafs_ is None

    def has_leaves(self):
        return self._leafs_ is not None and len(self._leafs_) > 0

    @property
    def leafs_(self):
        if self._leafs_ is None:
            raise RuntimeError("NgramPart was not initialized.")
        return self._leafs_

    def find(self, key):
        if not self.has_leaves():
            return None
        if key in self._leafs_:  # type: ignore
            return key
        return None

    def emplace(self, key, value):
        return self.leafs_.emplace(key, value)

    def __getitem__(self, key):
        return self._leafs_[key]  # type: ignore


class WeightingCriteria(IntEnum):
    NONE = 0
    TF = 1
    IDF = 2
    TFIDF = 3


def populate_grams(
    els,
    els_index,
    n_ngrams: int,
    ngram_size: int,
    ngram_id: int,
    c,  # : ForwardIter ,  # Map
):
    for _ngrams in range(n_ngrams, 0, -1):
        n = 1
        m = c
        while els_index < len(els):
            p = m.emplace(els[els_index], NgramPart(0))
            if n == ngram_size:
                p.id_ = ngram_id
                ngram_id += 1
                els_index += 1
                break
            if p.empty():
                p.init()
            m = p.leafs_
            n += 1
            els_index += 1
    return ngram_id


class TfIdfVectorizer(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        mode = self.mode  # type: ignore

        if mode == "TF":
            self.weighting_criteria_ = WeightingCriteria.TF
        elif mode == "IDF":
            self.weighting_criteria_ = WeightingCriteria.IDF
        elif mode == "TFIDF":
            self.weighting_criteria_ = WeightingCriteria.TFIDF

        self.min_gram_length_ = self.min_gram_length  # type: ignore
        self.max_gram_length_ = self.max_gram_length  # type: ignore
        self.max_skip_count_ = self.max_skip_count  # type: ignore
        self.ngram_counts_ = self.ngram_counts  # type: ignore
        self.max_gram_length_ = self.max_gram_length  # type: ignore
        self.ngram_indexes_ = self.ngram_indexes  # type: ignore
        self.output_size_ = max(self.ngram_indexes_) + 1
        self.weights_ = self.weights  # type: ignore
        self.pool_int64s_ = self.pool_int64s  # type: ignore

        self.int64_map_ = NgramPart(-10)
        self.int64_map_.init()

        total_items = len(self.pool_int64s_)
        ngram_id = 1  # start with 1, 0 - means no n-gram
        # Load into dictionary only required gram sizes
        ngram_size = 1
        for i in range(len(self.ngram_counts_)):
            start_idx = self.ngram_counts_[i]
            end_idx = (
                self.ngram_counts_[i + 1]
                if (i + 1) < len(self.ngram_counts_)
                else total_items
            )
            items = end_idx - start_idx
            if items > 0:
                ngrams = items // ngram_size
                if (
                    ngram_size >= self.min_gram_length_
                    and ngram_size <= self.max_gram_length_
                ):
                    ngram_id = populate_grams(
                        self.pool_int64s_,
                        start_idx,
                        ngrams,
                        ngram_size,
                        ngram_id,
                        self.int64_map_,
                    )
                else:
                    ngram_id += ngrams
            ngram_size += 1

    def increment_count(
        self, ngram_id: int, row_num: int, frequencies: List[int]
    ) -> None:
        ngram_id -= 1
        # assert(ngram_id < ngram_indexes_.size());
        output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
        # assert(static_cast<size_t>(output_idx) < frequencies.size());
        frequencies[output_idx] += 1

    def output_result(self, B: int, frequencies: List[int]) -> np.ndarray:
        l_output_dims: List[int] = []
        if B == 0:
            l_output_dims.append(self.output_size_)
            B = 1
        else:
            l_output_dims.append(B)
            l_output_dims.append(self.output_size_)
        output_dims = tuple(l_output_dims)

        row_size = self.output_size_

        total_dims = np.prod(output_dims)
        Y = np.empty((total_dims,), dtype=np.float32)

        w = self.weights_
        if self.weighting_criteria_ == WeightingCriteria.TF:
            i = 0
            for f in frequencies:
                Y[i] = f
                i += 1
        elif self.weighting_criteria_ == WeightingCriteria.IDF:
            if len(w) > 0:
                p = 0
                for _batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] if frequencies[p] > 0 else 0
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = 1 if f > 0 else 0
                    p += 1
        elif self.weighting_criteria_ == WeightingCriteria.TFIDF:
            if len(w) > 0:
                p = 0
                for _batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] * frequencies[p]
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = f
                    p += 1
        else:
            raise RuntimeError("Unexpected weighting_criteria.")
        return Y.reshape(output_dims)

    def compute_impl(  # type: ignore
        self,
        X: np.ndarray,
        row_num: int,
        row_size: int,
        frequencies: List[int],
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ) -> None:
        if len(X.shape) > 1:
            X_flat = X[row_num]
        else:
            X_flat = X
        row_begin = 0
        row_end = row_begin + row_size

        max_skip_distance = max_skip_count + 1
        start_ngram_size = min_gram_length

        for skip_distance in range(1, max_skip_distance + 1):
            ngram_start = row_begin
            ngram_row_end = row_end

            while ngram_start < ngram_row_end:
                # We went far enough so no n-grams of any size can be gathered
                at_least_this = ngram_start + skip_distance * (start_ngram_size - 1)
                if at_least_this >= ngram_row_end:
                    break

                ngram_item = ngram_start
                int_map = self.int64_map_
                ngram_size = 1
                while (
                    int_map.has_leaves()
                    and ngram_size <= max_gram_length
                    and ngram_item < ngram_row_end
                ):
                    val = X_flat[ngram_item]
                    hit = int_map.find(val)
                    if hit is None:
                        break
                    hit = int_map[val].id_
                    if ngram_size >= start_ngram_size and hit != 0:
                        self.increment_count(hit, row_num, frequencies)
                    int_map = int_map[val]
                    ngram_size += 1
                    ngram_item += skip_distance

                ngram_start += 1

            # We count UniGrams only once since they are not affected by skip_distance
            if start_ngram_size == 1:
                start_ngram_size += 1
                if start_ngram_size > max_gram_length:
                    break

    def _run(  # type: ignore
        self,
        X,
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ):
        # weights should be identical to self.weights as well as
        # pool_strings, pool_int64s, ngram_indexes, ngram_counts, mode.
        # This means none of those attributes can be used in one function.

        total_items = np.prod(X.shape)

        num_rows = 0
        B = 0
        C = 0
        input_dims = X.shape
        if len(input_dims) == 0:
            num_rows = 1
            C = 1
            if total_items != 1:
                raise ValueError(f"Unexpected total of items {total_items}.")
        elif len(input_dims) == 1:
            num_rows = 1
            C = input_dims[0]
        elif len(input_dims) == 2:
            B = input_dims[0]
            C = input_dims[1]
            num_rows = B
            if B < 1:
                raise ValueError(
                    f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
                )
        else:
            raise ValueError(
                f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
            )

        if num_rows * C != total_items:
            raise ValueError(
                f"Unexpected total of items, num_rows * C = {num_rows * C} != total_items = {total_items}."
            )
        # Frequency holder allocate [B..output_size_] and init all to zero
        frequencies = np.zeros((num_rows * self.output_size_,), dtype=np.int64)

        if total_items == 0 or self.int64_map_.empty():
            # TfidfVectorizer may receive an empty input when it follows a Tokenizer
            # (for example for a string containing only stopwords).
            # TfidfVectorizer returns a zero tensor of shape
            # {b_dim, output_size} when b_dim is the number of received observations
            # and output_size the is the maximum value in ngram_indexes attribute plus 1.
            return self.output_result(B, frequencies)  # type: ignore[arg-type]

        def fn(row_num):
            self.compute_impl(
                X,
                row_num,
                C,
                frequencies,  # type: ignore[arg-type]
                max_gram_length=max_gram_length,
                max_skip_count=max_skip_count,
                min_gram_length=min_gram_length,
                mode=mode,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s,
                pool_strings=pool_strings,
                weights=weights,
            )

        # can be parallelized.
        for i in range(num_rows):
            fn(i)

        return (self.output_result(B, frequencies),)  # type: ignore[arg-type]
