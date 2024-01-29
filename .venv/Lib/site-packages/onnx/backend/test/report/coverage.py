# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import csv
import datetime
import os
from collections import OrderedDict, defaultdict
from typing import IO, Any, Dict, List, Optional, Set

from tabulate import tabulate

import onnx
from onnx import GraphProto, defs, helper

_all_schemas = defs.get_all_schemas()


class AttrCoverage:
    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.values: Set[str] = set()

    def add(self, attr: onnx.AttributeProto) -> None:
        assert self.name in {None, attr.name}
        self.name = attr.name
        value = helper.get_attribute_value(attr)
        # Turn list into tuple so we can put it into set
        # As value can be string, don't blindly turn `collections.Iterable`
        # into tuple.
        if isinstance(value, list):
            value = tuple(value)
        self.values.add(str(value))


class NodeCoverage:
    def __init__(self) -> None:
        self.op_type: Optional[str] = None
        self.attr_coverages: Dict[str, AttrCoverage] = defaultdict(AttrCoverage)

    def add(self, node: onnx.NodeProto) -> None:
        assert self.op_type in [None, node.op_type]

        if self.op_type is None:
            self.op_type = node.op_type
            assert self.op_type is not None
            self.schema = defs.get_schema(self.op_type, domain=node.domain)

        for attr in node.attribute:
            self.attr_coverages[attr.name].add(attr)


class ModelCoverage:
    def __init__(self) -> None:
        self.name: Optional[str] = None
        self.graph: Optional[GraphProto] = None
        self.node_coverages: Dict[str, NodeCoverage] = defaultdict(NodeCoverage)

    def add(self, model: onnx.ModelProto) -> None:
        assert self.name in [None, model.graph.name]

        if self.name is None:
            self.name = model.graph.name
            assert self.name is not None
            self.graph = model.graph

        for node in model.graph.node:
            self.node_coverages[node.op_type].add(node)


class Coverage:
    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, NodeCoverage]] = {
            "loaded": defaultdict(NodeCoverage),
            "passed": defaultdict(NodeCoverage),
        }
        self.models: Dict[str, Dict[str, ModelCoverage]] = {
            "loaded": defaultdict(ModelCoverage),
            "passed": defaultdict(ModelCoverage),
        }

    def add_node(self, node: onnx.NodeProto, bucket: str) -> None:
        self.buckets[bucket][node.op_type].add(node)

    def add_graph(self, graph: onnx.GraphProto, bucket: str) -> None:
        for node in graph.node:
            self.add_node(node, bucket)

    def add_model(self, model: onnx.ModelProto, bucket: str, is_model: bool) -> None:
        self.add_graph(model.graph, bucket)
        # Only add model if name does not start with test
        if is_model:
            self.models[bucket][model.graph.name].add(model)

    def add_proto(self, proto: onnx.ModelProto, bucket: str, is_model: bool) -> None:
        assert isinstance(proto, onnx.ModelProto)
        self.add_model(proto, bucket, is_model)

    def report_text(self, writer: IO[str]) -> None:
        writer.write("---------- onnx coverage: ----------\n")
        writer.write(
            f"Operators (passed/loaded/total): {len(self.buckets['passed'])}/{len(self.buckets['loaded'])}/{len(_all_schemas)}\n"
        )
        writer.write("------------------------------------\n")

        rows = []
        passed = []
        all_ops: List[str] = []
        experimental: List[str] = []
        for op_cov in self.buckets["passed"].values():
            covered_attrs = [
                f"{attr_cov.name}: {len(attr_cov.values)}"
                for attr_cov in op_cov.attr_coverages.values()
            ]
            uncovered_attrs = [
                f"{attr}: 0"
                for attr in op_cov.schema.attributes
                if attr not in op_cov.attr_coverages
            ]
            attrs = sorted(covered_attrs) + sorted(uncovered_attrs)
            if attrs:
                attrs_column = os.linesep.join(attrs)
            else:
                attrs_column = "No attributes"
            rows.append([op_cov.op_type, attrs_column])
            passed.append(op_cov.op_type)
        writer.write(
            tabulate(
                rows,
                headers=["Operator", "Attributes\n(name: #values)"],
                tablefmt="plain",
            )
        )
        writer.write("\n")
        if os.environ.get("CSVDIR") is not None:
            self.report_csv(all_ops, passed, experimental)

    # This function writes the coverage report to a set of CSV files for
    # the Backend Scoreboard (onnx.ai/backend-scoreboard). To enable this
    # feature, set a CSVDIR environment variable locally with the directory
    # where you would like the files to be written, relative to the
    # directory from which you're running pytest.  The format of the CSV
    # files is a column naming each op or model and columns for each
    # backend with indications of whether the tests passed or failed for
    # each row.
    def report_csv(
        self, all_ops: List[str], passed: List[Optional[str]], experimental: List[str]
    ) -> None:
        for schema in _all_schemas:
            if schema.domain in {"", "ai.onnx"}:
                all_ops.append(schema.name)
                if schema.support_level == defs.OpSchema.SupportType.EXPERIMENTAL:
                    experimental.append(schema.name)
        all_ops.sort()
        nodes_path = os.path.join(
            str(os.environ.get("CSVDIR")), "nodes.csv"  # type: ignore
        )  # type: ignore
        models_path = os.path.join(
            str(os.environ.get("CSVDIR")), "models.csv"  # type: ignore
        )  # type: ignore
        existing_nodes: OrderedDict[str, Dict[str, str]] = OrderedDict()
        existing_models: OrderedDict[str, Dict[str, str]] = OrderedDict()
        frameworks: List[str] = []
        if os.path.isfile(nodes_path):
            with open(nodes_path) as nodes_file:
                reader = csv.DictReader(nodes_file)
                assert reader.fieldnames
                frameworks = list(reader.fieldnames)
                for row in reader:
                    op = row["Op"]
                    del row["Op"]
                    existing_nodes[str(op)] = row
        if os.path.isfile(models_path):
            with open(models_path) as models_file:
                reader = csv.DictReader(models_file)
                for row in reader:
                    model = row["Model"]
                    del row["Model"]
                    existing_models[str(model)] = row
        backend = os.environ.get("BACKEND")
        other_frameworks = frameworks[1:]
        with open(nodes_path, "w") as nodes_file:
            if "Op" not in frameworks:
                frameworks.append("Op")
            if backend not in frameworks:
                frameworks.append(str(backend))
            else:
                other_frameworks.remove(str(backend))
            node_writer = csv.DictWriter(nodes_file, fieldnames=frameworks)
            node_writer.writeheader()
            for node in all_ops:
                node_name = node
                if node in experimental:
                    node_name = node + " (Experimental)"
                if node_name not in existing_nodes:
                    # Also add Skipped for other nodes
                    existing_nodes[node_name] = OrderedDict()
                    for other_framework in other_frameworks:
                        existing_nodes[node_name][other_framework] = "Skipped!"
                if node in passed:
                    existing_nodes[node_name][str(backend)] = "Passed!"
                else:
                    existing_nodes[node_name][str(backend)] = "Failed!"
            summaries: Dict[Any, Any] = {}
            if "Summary" in existing_nodes:
                summaries = existing_nodes["Summary"]
                del existing_nodes["Summary"]
            summaries[str(backend)] = f"{len(passed)}/{len(all_ops)} node tests passed"
            summaries["Op"] = "Summary"
            for node in existing_nodes:
                existing_nodes[node]["Op"] = str(node)
                node_writer.writerow(existing_nodes[node])
            node_writer.writerow(summaries)
        with open(models_path, "w") as models_file:
            frameworks[0] = "Model"
            model_writer = csv.DictWriter(models_file, fieldnames=frameworks)
            model_writer.writeheader()
            # Consider both buckets
            num_models = 0
            for bucket in self.models:
                for model in self.models[bucket]:  # type: ignore
                    # Both analyze and run the model on the backend
                    num_covered = 0
                    for node in self.models[bucket][model].node_coverages:
                        if node in passed:
                            num_covered += 1
                    # TODO: Identify if there are models that are being
                    # skipped/not loaded, but that are in other frameworks
                    msg = "Passed!"
                    if bucket == "loaded":
                        if model in self.models["passed"]:
                            continue
                        msg = "Failed!"
                    num_models += 1
                    if model not in existing_models:
                        # Also add Skipped for other models
                        existing_models[model] = OrderedDict()
                        for other_framework in other_frameworks:
                            existing_models[model][other_framework] = "Skipped!"
                    existing_models[model][str(backend)] = str(
                        f"{num_covered}/{len(self.models[bucket][model].node_coverages)} nodes covered: {msg}"
                    )
            summaries.clear()
            if "Summary" in existing_models:
                summaries = existing_models["Summary"]
                del existing_models["Summary"]
            if str(backend) in summaries:
                del summaries[str(backend)]
            summaries[
                str(backend)
            ] = f"{len(self.models['passed'])}/{num_models} model tests passed"
            summaries["Model"] = "Summary"
            for model in existing_models:  # type: ignore
                existing_models[model]["Model"] = model
                model_writer.writerow(existing_models[model])
            model_writer.writerow(summaries)
        with open(
            os.path.join(str(os.environ.get("CSVDIR")), "metadata.csv"),  # type: ignore
            "w",
        ) as metadata_file:  # type: ignore
            metadata_writer = csv.writer(metadata_file)
            metadata_writer.writerow(
                ["Latest Update", datetime.datetime.now().isoformat().replace("T", " ")]
            )
