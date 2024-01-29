# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


class Scan(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if not hasattr(self.body, "run"):  # type: ignore
            raise RuntimeError(
                f"Parameter 'body' must have a method 'run', type {type(self.body)}."  # type: ignore
            )
        self.input_directions_ = [
            0
            if self.scan_input_directions is None  # type: ignore
            or i >= len(self.scan_input_directions)  # type: ignore
            else self.scan_input_directions[i]  # type: ignore
            for i in range(self.num_scan_inputs)  # type: ignore
        ]
        max_dir_in = max(self.input_directions_)
        if max_dir_in != 0:
            raise RuntimeError(
                "Scan is not implemented for other output input_direction than 0."
            )
        self.input_axes_ = [
            0
            if self.scan_input_axes is None or i >= len(self.scan_input_axes)  # type: ignore
            else self.scan_input_axes[i]  # type: ignore
            for i in range(self.num_scan_inputs)  # type: ignore
        ]
        max_axe_in = max(self.input_axes_)
        if max_axe_in != 0:
            raise RuntimeError("Scan is not implemented for other input axes than 0.")
        self.input_names = self.body.input_names  # type: ignore
        self.output_names = self.body.output_names  # type: ignore

    def _common_run_shape(self, *args):  # type: ignore
        num_loop_state_vars = len(args) - self.num_scan_inputs  # type: ignore
        num_scan_outputs = len(args) - num_loop_state_vars

        output_directions = [
            0
            if self.scan_output_directions is None  # type: ignore
            or i >= len(self.scan_output_directions)  # type: ignore
            else self.scan_output_directions[i]  # type: ignore
            for i in range(num_scan_outputs)
        ]
        max_dir_out = max(output_directions)
        if max_dir_out != 0:
            raise RuntimeError(
                "Scan is not implemented for other output output_direction than 0."
            )
        output_axes = [
            0
            if self.scan_output_axes is None or i >= len(self.scan_output_axes)  # type: ignore
            else self.scan_output_axes[i]  # type: ignore
            for i in range(num_scan_outputs)
        ]
        max_axe_out = max(output_axes)
        if max_axe_out != 0:
            raise RuntimeError("Scan is not implemented for other output axes than 0.")

        state_names_in = self.input_names[: self.num_scan_inputs]  # type: ignore
        state_names_out = self.output_names[: len(state_names_in)]
        scan_names_in = self.input_names[num_loop_state_vars:]
        scan_names_out = self.output_names[num_loop_state_vars:]
        scan_values = args[num_loop_state_vars:]

        states = args[:num_loop_state_vars]

        return (
            num_loop_state_vars,
            num_scan_outputs,
            output_directions,
            max_dir_out,
            output_axes,
            max_axe_out,
            state_names_in,
            state_names_out,
            scan_names_in,
            scan_names_out,
            scan_values,
            states,
        )

    def _run(  # type:ignore
        self,
        *args,
        body=None,
        num_scan_inputs=None,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
        attributes=None,
    ):
        # TODO: support overridden attributes.
        (
            num_loop_state_vars,
            num_scan_outputs,
            output_directions,
            max_dir_out,
            output_axes,
            max_axe_out,
            state_names_in,
            state_names_out,
            scan_names_in,
            scan_names_out,
            scan_values,
            states,
        ) = self._common_run_shape(*args)

        max_iter = args[num_loop_state_vars].shape[self.input_axes_[0]]
        results = [[] for _ in scan_names_out]  # type: ignore

        for it in range(max_iter):
            inputs = {}
            for name, value in zip(state_names_in, states):
                inputs[name] = value
            for name, value in zip(scan_names_in, scan_values):
                inputs[name] = value[it]

            try:
                outputs_list = self._run_body(inputs)  # type: ignore
            except TypeError as e:
                raise TypeError(
                    f"Unable to call 'run' for type '{type(self.body)}'."  # type: ignore
                ) from e

            outputs = dict(zip(self.output_names, outputs_list))
            states = [outputs[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                results[i].append(np.expand_dims(outputs[name], axis=0))

        for res in results:
            conc = np.vstack(res)
            states.append(conc)
        return tuple(states)
