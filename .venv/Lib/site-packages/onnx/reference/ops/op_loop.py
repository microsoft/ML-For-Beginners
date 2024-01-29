# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Loop(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if "opsets" not in self.run_params:
            raise KeyError("run_params must contains key 'opsets'.")
        if "verbose" not in run_params:
            raise KeyError("run_params must contains key 'verbose'.")
        self.output_index = {n: i for i, n in enumerate(self.body.output_names)}  # type: ignore
        self.N = len(self.body.input_names) - 2  # type: ignore
        self.K = len(self.body.output_names) - self.N - 1  # type: ignore

    def need_context(self) -> bool:
        """
        The operator Loop needs to know all results produced
        so far as the loop may silently access one of them.
        Some information are not always referred in the list of inputs
        (kind of static variables).
        """
        return True

    def _run(self, M, cond, *args, context=None, body=None, attributes=None):  # type: ignore
        if args:
            v_initial = args[0]
            args = args[1:]
        else:
            v_initial = None
        if not hasattr(M, "dtype"):
            raise TypeError(f"M must be an array or a numpy number not {type(M)}.")
        body = self.body  # type: ignore
        loop_inputs = body.input_names
        inputs = {name: None for name in loop_inputs}
        if v_initial is not None:
            inputs[loop_inputs[2]] = v_initial
        cond_name = body.output_names[0]
        if args:
            begin = len(loop_inputs) - len(args)
            all_inputs = loop_inputs[begin:]
            for name, val in zip(all_inputs, args):
                inputs[name] = val
        if context is not None:
            for a in context:
                inputs[a] = context[a]

        k_carried_away = [[] for i in range(self.K)]  # type: ignore
        it = 0
        while cond and it < M:
            self._log("  -- loop> {%r}", context)
            if len(body.input_names) > 0 and body.input_names[0] is not None:
                inputs[body.input_names[0]] = np.array(it, dtype=M.dtype)  # type: ignore
            if len(body.input_names) > 1 and body.input_names[1] is not None:
                inputs[body.input_names[1]] = cond
            outputs = self._run_body(inputs, attributes=attributes)  # type: ignore
            if self.K > 0:
                for k in range(self.K):
                    k_carried_away[k].append(outputs[-self.K + k])
            index_cond = self.output_index[cond_name]
            cond = outputs[index_cond]
            if cond is None:
                raise RuntimeError(
                    f"Condition {cond_name!r} returned by the subgraph cannot be None."
                )
            for i, o in zip(body.input_names[2:], body.output_names[1:]):
                inputs[i] = outputs[self.output_index[o]]
            it += 1
            self._log("  -- loop<")

        if it == 0:
            outputs = [inputs[i] for i in body.input_names[2:]]
        else:
            outputs = outputs[1 : 1 + self.N]
        outputs.extend(k_carried_away)
        while len(outputs) < len(self.onnx_node.output):
            outputs.append(np.empty(shape=()))
        res = tuple(outputs)

        return res
