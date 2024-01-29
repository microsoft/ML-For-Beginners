# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class If(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if "opsets" not in self.run_params:
            raise KeyError("run_params must contains key 'opsets'.")
        if "verbose" not in run_params:
            raise KeyError("run_params must contains key 'verbose'.")

    def need_context(self) -> bool:
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        """
        return True

    def _run(
        self,
        cond: np.ndarray | np.bool_,
        context=None,
        else_branch=None,
        then_branch=None,
        attributes=None,
    ):
        if cond.size != 1:
            raise ValueError(
                f"Operator If ({self.onnx_node.name!r}) expects a single element as condition, but the size of 'cond' is {len(cond)}."
            )
        cond_ = cond.item(0)
        if cond_:
            self._log("  -- then> {%r}", context)
            outputs = self._run_then_branch(context, attributes=attributes)  # type: ignore
            self._log("  -- then<")
            final = tuple(outputs)
            branch = "then"
        else:
            self._log("  -- else> {%r}", context)
            outputs = self._run_else_branch(context, attributes=attributes)  # type: ignore
            self._log("  -- else<")
            final = tuple(outputs)
            branch = "else"

        if not final:
            raise RuntimeError(
                f"Operator If ({self.onnx_node.name!r}) does not have any output."
            )
        for i, f in enumerate(final):
            if f is None:
                br = self.then_branch if branch == "then" else self.else_branch  # type: ignore
                names = br.output_names
                inits = [i.name for i in br.obj.graph.initializer]
                raise RuntimeError(
                    f"Output {i!r} (branch={branch!r}, name={names[i]!r}) is None, "
                    f"available inputs={sorted(context)}, initializers={inits}."
                )
        return final
