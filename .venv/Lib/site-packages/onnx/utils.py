# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Tuple

import onnx.checker
import onnx.helper
import onnx.shape_inference
from onnx import FunctionProto, ModelProto, NodeProto, TensorProto, ValueInfoProto


class Extractor:
    def __init__(self, model: ModelProto) -> None:
        self.model = onnx.shape_inference.infer_shapes(model)
        self.graph = self.model.graph
        self.wmap = self._build_name2obj_dict(self.graph.initializer)
        self.vimap = self._build_name2obj_dict(self.graph.value_info)

    @staticmethod
    def _build_name2obj_dict(objs):  # type: ignore
        return {obj.name: obj for obj in objs}

    def _collect_new_io_core(self, original_io, io_names_to_extract):  # type: ignore
        original_io_map = self._build_name2obj_dict(original_io)
        original_io_names = set(original_io_map)
        s_io_names_to_extract = set(io_names_to_extract)
        io_names_to_keep = s_io_names_to_extract & original_io_names
        new_io_names_to_add = s_io_names_to_extract - original_io_names

        new_io_tensors = []
        for name in io_names_to_keep:
            new_io_tensors.append(original_io_map[name])
        for name in new_io_names_to_add:
            # activation become input or output
            new_io_tensors.append(self.vimap[name])

        # adjust sequence
        new_io_tensors_map = self._build_name2obj_dict(new_io_tensors)
        return [new_io_tensors_map[name] for name in io_names_to_extract]

    def _collect_new_inputs(self, names: List[str]) -> List[ValueInfoProto]:
        return self._collect_new_io_core(self.graph.input, names)  # type: ignore

    def _collect_new_outputs(self, names: List[str]) -> List[ValueInfoProto]:
        return self._collect_new_io_core(self.graph.output, names)  # type: ignore

    def _dfs_search_reachable_nodes(
        self,
        node_output_name: str,
        graph_input_names: List[str],
        reachable_nodes: List[NodeProto],
    ) -> None:
        if node_output_name in graph_input_names:
            return
        for node in self.graph.node:
            # check output_name first to reduce run time
            if node_output_name not in node.output:
                continue
            if node in reachable_nodes:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                self._dfs_search_reachable_nodes(
                    name, graph_input_names, reachable_nodes
                )

    def _collect_reachable_nodes(
        self,
        input_names: List[str],
        output_names: List[str],
    ) -> List[NodeProto]:
        reachable_nodes = []  # type: ignore[var-annotated]
        for name in output_names:
            self._dfs_search_reachable_nodes(name, input_names, reachable_nodes)
        # needs to be topology sorted.
        nodes = [n for n in self.graph.node if n in reachable_nodes]
        return nodes

    def _collect_referred_local_functions(
        self,
        nodes,  # type: List[NodeProto]
    ):  # type: (...) -> List[FunctionProto]
        # a node in a model graph may refer a function.
        # a function contains nodes, some of which may in turn refer a function.
        # we need to find functions referred by graph nodes and
        # by nodes used to define functions.
        def find_referred_funcs(nodes, referred_local_functions):  # type: ignore
            new_nodes = []  # type: List[NodeProto]
            for node in nodes:
                # check if the node is a function op
                match_function = next(
                    (
                        f
                        for f in self.model.functions
                        if f.name == node.op_type and f.domain == node.domain
                    ),
                    None,
                )
                if match_function and match_function not in referred_local_functions:
                    referred_local_functions.append(match_function)
                    new_nodes.extend(match_function.node)

            return new_nodes

        referred_local_functions = []  # type: List[FunctionProto]
        new_nodes = find_referred_funcs(nodes, referred_local_functions)
        while new_nodes:
            new_nodes = find_referred_funcs(new_nodes, referred_local_functions)

        return referred_local_functions

    def _collect_reachable_tensors(
        self,
        nodes: List[NodeProto],
    ) -> Tuple[List[TensorProto], List[ValueInfoProto]]:
        all_tensors_name = set()
        for node in nodes:
            for name in node.input:
                all_tensors_name.add(name)
            for name in node.output:
                all_tensors_name.add(name)

        initializer = [self.wmap[t] for t in self.wmap if t in all_tensors_name]
        value_info = [self.vimap[t] for t in self.vimap if t in all_tensors_name]
        len_sparse_initializer = len(self.graph.sparse_initializer)
        if len_sparse_initializer != 0:
            raise ValueError(
                f"len_sparse_initializer is {len_sparse_initializer}, it must be 0."
            )
        len_quantization_annotation = len(self.graph.quantization_annotation)
        if len_quantization_annotation != 0:
            raise ValueError(
                f"len_quantization_annotation is {len_quantization_annotation}, it must be 0."
            )
        return initializer, value_info

    def _make_model(
        self,
        nodes: List[NodeProto],
        inputs: List[ValueInfoProto],
        outputs: List[ValueInfoProto],
        initializer: List[TensorProto],
        value_info: List[ValueInfoProto],
        local_functions: List[FunctionProto],
    ) -> ModelProto:
        name = "Extracted from {" + self.graph.name + "}"
        graph = onnx.helper.make_graph(
            nodes, name, inputs, outputs, initializer=initializer, value_info=value_info
        )

        meta = {
            "ir_version": self.model.ir_version,
            "opset_imports": self.model.opset_import,
            "producer_name": "onnx.utils.extract_model",
            "functions": local_functions,
        }
        return onnx.helper.make_model(graph, **meta)

    def extract_model(
        self,
        input_names: List[str],
        output_names: List[str],
    ) -> ModelProto:
        inputs = self._collect_new_inputs(input_names)
        outputs = self._collect_new_outputs(output_names)
        nodes = self._collect_reachable_nodes(input_names, output_names)
        initializer, value_info = self._collect_reachable_tensors(nodes)
        local_functions = self._collect_referred_local_functions(nodes)
        model = self._make_model(
            nodes, inputs, outputs, initializer, value_info, local_functions
        )

        return model


def extract_model(
    input_path: str,
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    check_model: bool = True,
) -> None:
    """Extracts sub-model from an ONNX model.

    The sub-model is defined by the names of the input and output tensors *exactly*.

    Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
    which is defined by the input and output tensors, should not _cut through_ the
    subgraph that is connected to the _main graph_ as attributes of these operators.

    Arguments:
        input_path (string): The path to original ONNX model.
        output_path (string): The path to save the extracted ONNX model.
        input_names (list of string): The names of the input tensors that to be extracted.
        output_names (list of string): The names of the output tensors that to be extracted.
        check_model (bool): Whether to run model checker on the extracted model.
    """
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid input model path: {input_path}")
    if not output_path:
        raise ValueError("Output model path shall not be empty!")
    if not output_names:
        raise ValueError("Output tensor names shall not be empty!")

    onnx.checker.check_model(input_path)
    model = onnx.load(input_path)

    e = Extractor(model)
    extracted = e.extract_model(input_names, output_names)

    onnx.save(extracted, output_path)
    if check_model:
        onnx.checker.check_model(output_path)
