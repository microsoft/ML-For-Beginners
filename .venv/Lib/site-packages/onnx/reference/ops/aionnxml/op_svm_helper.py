# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Any

import numpy as np


class SVMAttributes:
    def __init__(self):
        self._names = []

    def add(self, name: str, value: Any) -> None:
        if isinstance(value, list) and name not in {"kernel_params"}:
            if name in {"vectors_per_class"}:
                value = np.array(value, dtype=np.int64)
            else:
                value = np.array(value, dtype=np.float32)
        setattr(self, name, value)

    def __str__(self) -> str:
        rows = ["Attributes"]
        for name in self._names:
            rows.append(f"  {name}={getattr(self, name)}")
        return "\n".join(rows)


class SVMCommon:
    """
    Base class for SVM.
    """

    def __init__(self, **kwargs):  # type: ignore
        self.atts = SVMAttributes()

        for name, value in kwargs.items():
            self.atts.add(name, value)

        if self.atts.kernel_params:  # type: ignore
            self.gamma_ = self.atts.kernel_params[0]  # type: ignore
            self.coef0_ = self.atts.kernel_params[1]  # type: ignore
            self.degree_ = int(self.atts.kernel_params[2])  # type: ignore
        else:
            self.gamma_ = 0.0
            self.coef0_ = 0.0
            self.degree_ = 0

    def __str__(self) -> str:
        rows = ["TreeEnsemble", f"root_index={self.root_index}", str(self.atts)]  # type: ignore
        return "\n".join(rows)

    def kernel_dot(self, pA: np.ndarray, pB: np.ndarray, kernel: str) -> np.ndarray:
        k = kernel.lower()
        if k == "poly":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return s**self.degree_  # type: ignore
        if k == "sigmoid":
            s = np.dot(pA, pB)
            s = s * self.gamma_ + self.coef0_
            return np.tanh(s)  # type: ignore
        if k == "rbf":
            diff = pA - pB
            s = (diff * diff).sum()
            return np.exp(-self.gamma_ * s)  # type: ignore
        if k == "linear":
            return np.dot(pA, pB)  # type: ignore
        raise ValueError(f"Unexpected kernel={kernel!r}.")

    def run_reg(self, X: np.ndarray) -> np.ndarray:
        if self.atts.n_supports > 0:  # type: ignore
            # length of each support vector
            mode_ = "SVM_SVC"
            kernel_type_ = self.atts.kernel_type  # type: ignore
            sv = self.atts.support_vectors.reshape((self.atts.n_supports, -1))  # type: ignore
        else:
            mode_ = "SVM_LINEAR"
            kernel_type_ = "LINEAR"

        z = np.empty((X.shape[0], 1), dtype=X.dtype)
        for n in range(X.shape[0]):
            s = 0.0

            if mode_ == "SVM_SVC":
                for j in range(self.atts.n_supports):  # type: ignore
                    d = self.kernel_dot(X[n], sv[j], kernel_type_)
                    s += self.atts.coefficients[j] * d  # type: ignore
                s += self.atts.rho[0]  # type: ignore
            elif mode_ == "SVM_LINEAR":
                s = self.kernel_dot(X[n], self.atts.coefficients, kernel_type_)  # type: ignore
                s += self.atts.rho[0]  # type: ignore

            if self.atts.one_class:  # type: ignore
                z[n, 0] = 1 if s > 0 else -1
            else:
                z[n, 0] = s
        return z
