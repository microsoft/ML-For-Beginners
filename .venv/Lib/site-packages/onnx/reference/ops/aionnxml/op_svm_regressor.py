# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon


class SVMRegressor(OpRunAiOnnxMl):
    """
    The class only implements `POST_TRANSFORM="NONE"`.
    """

    def _run(  # type: ignore
        self,
        X,
        coefficients=None,
        kernel_params=None,
        kernel_type=None,
        n_targets=None,
        n_supports=None,
        one_class=None,
        post_transform=None,
        rho=None,
        support_vectors=None,
    ):
        svm = SVMCommon(
            coefficients=coefficients,
            kernel_params=kernel_params,
            kernel_type=kernel_type,
            n_targets=n_targets,
            n_supports=n_supports,
            one_class=one_class,
            post_transform=post_transform,
            rho=rho,
            support_vectors=support_vectors,
        )
        # adding an attribute for debugging purpose
        self._svm = svm
        res = svm.run_reg(X)

        if post_transform in (None, "NONE"):
            return (res,)
        raise NotImplementedError(f"post_transform={post_transform!r} not implemented.")
