#!/usr/bin/env python

# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Sequence, Set, Tuple

import numpy as np

from onnx import defs, helper
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets
from onnx.defs import ONNX_ML_DOMAIN, OpSchema

SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv("ONNX_ML") == "0")


def display_number(v: int) -> str:
    if defs.OpSchema.is_infinite(v):
        return "&#8734;"
    return str(v)


def should_render_domain(domain: str, output: str) -> bool:
    is_ml = "-ml" in output
    if domain == ONNX_ML_DOMAIN:
        return is_ml
    else:
        return not is_ml


def format_name_with_domain(domain: str, schema_name: str) -> str:
    if domain:
        return f"{domain}.{schema_name}"
    return schema_name


def format_function_versions(function_versions: Sequence[int]) -> str:
    return f"{', '.join([str(v) for v in function_versions])}"


def format_versions(versions: Sequence[OpSchema], changelog: str) -> str:
    return f"{', '.join(display_version_link(format_name_with_domain(v.domain, v.name), v.since_version, changelog) for v in versions[::-1])}"


def display_attr_type(v: OpSchema.AttrType) -> str:
    assert isinstance(v, OpSchema.AttrType)
    s = str(v)
    s = s[s.rfind(".") + 1 :].lower()
    if s[-1] == "s":
        s = "list of " + s
    return s


def display_domain(domain: str) -> str:
    if domain:
        return f"the '{domain}' operator set"
    return "the default ONNX operator set"


def display_domain_short(domain: str) -> str:
    if domain:
        return domain
    return "ai.onnx (default)"


def display_version_link(name: str, version: int, changelog: str) -> str:
    name_with_ver = f"{name}-{version}"
    return f'<a href="{changelog}#{name_with_ver}">{version}</a>'


def generate_formal_parameter_tags(formal_parameter: OpSchema.FormalParameter) -> str:
    tags: List[str] = []
    if OpSchema.FormalParameterOption.Optional == formal_parameter.option:
        tags = ["optional"]
    elif OpSchema.FormalParameterOption.Variadic == formal_parameter.option:
        if formal_parameter.is_homogeneous:
            tags = ["variadic"]
        else:
            tags = ["variadic", "heterogeneous"]
    differentiable: OpSchema.DifferentiationCategory = (
        OpSchema.DifferentiationCategory.Differentiable
    )
    non_differentiable: OpSchema.DifferentiationCategory = (
        OpSchema.DifferentiationCategory.NonDifferentiable
    )
    if differentiable == formal_parameter.differentiation_category:
        tags.append("differentiable")
    elif non_differentiable == formal_parameter.differentiation_category:
        tags.append("non-differentiable")

    return "" if len(tags) == 0 else " (" + ", ".join(tags) + ")"


def display_schema(
    schema: OpSchema, versions: Sequence[OpSchema], changelog: str
) -> str:
    s = ""

    # doc
    if schema.doc:
        s += "\n"
        s += "\n".join(
            ("  " + line).rstrip() for line in schema.doc.lstrip().splitlines()
        )
        s += "\n"

    # since version
    s += "\n#### Version\n"
    if schema.support_level == OpSchema.SupportType.EXPERIMENTAL:
        s += "\nNo versioning maintained for experimental ops."
    else:
        s += (
            "\nThis version of the operator has been "
            + ("deprecated" if schema.deprecated else "available")
            + f" since version {schema.since_version}"
        )
        s += f" of {display_domain(schema.domain)}.\n"
        if len(versions) > 1:
            # TODO: link to the Changelog.md
            s += "\nOther versions of this operator: {}\n".format(
                ", ".join(
                    display_version_link(
                        format_name_with_domain(v.domain, v.name),
                        v.since_version,
                        changelog,
                    )
                    for v in versions[:-1]
                )
            )

    # If this schema is deprecated, don't display any of the following sections
    if schema.deprecated:
        return s

    # attributes
    if schema.attributes:
        s += "\n#### Attributes\n\n"
        s += "<dl>\n"
        for _, attr in sorted(schema.attributes.items()):
            # option holds either required or default value
            opt = ""
            if attr.required:
                opt = "required"
            elif attr.default_value.name:
                default_value = helper.get_attribute_value(attr.default_value)
                doc_string = attr.default_value.doc_string

                def format_value(value: Any) -> str:
                    if isinstance(value, float):
                        formatted = str(np.round(value, 5))
                        # use default formatting, unless too long.
                        if len(formatted) > 10:  # noqa: PLR2004
                            formatted = str(f"({value:e})")
                        return formatted
                    if isinstance(value, (bytes, bytearray)):
                        return str(value.decode("utf-8"))
                    return str(value)

                if isinstance(default_value, list):
                    default_value = [format_value(val) for val in default_value]
                else:
                    default_value = format_value(default_value)
                opt = f"default is {default_value}{doc_string}"

            s += f"<dt><tt>{attr.name}</tt> : {display_attr_type(attr.type)}{f' ({opt})' if opt else ''}</dt>\n"
            s += f"<dd>{attr.description}</dd>\n"
        s += "</dl>\n"

    # inputs
    s += "\n#### Inputs"
    if schema.min_input != schema.max_input:
        s += f" ({display_number(schema.min_input)} - {display_number(schema.max_input)})"
    s += "\n\n"
    if schema.inputs:
        s += "<dl>\n"
        for input_ in schema.inputs:
            option_str = generate_formal_parameter_tags(input_)
            s += f"<dt><tt>{input_.name}</tt>{option_str} : {input_.type_str}</dt>\n"
            s += f"<dd>{input_.description}</dd>\n"
        s += "</dl>\n"

    # outputs
    s += "\n#### Outputs"
    if schema.min_output != schema.max_output:
        s += f" ({display_number(schema.min_output)} - {display_number(schema.max_output)})"
    s += "\n\n"

    if schema.outputs:
        s += "<dl>\n"
        for output in schema.outputs:
            option_str = generate_formal_parameter_tags(output)
            s += f"<dt><tt>{output.name}</tt>{option_str} : {output.type_str}</dt>\n"
            s += f"<dd>{output.description}</dd>\n"
        s += "</dl>\n"

    # type constraints
    s += "\n#### Type Constraints"
    s += "\n\n"
    if schema.type_constraints:
        s += "<dl>\n"
        for type_constraint in schema.type_constraints:
            allowedTypes = type_constraint.allowed_type_strs
            if len(allowedTypes) > 0:
                allowedTypeStr = allowedTypes[0]
            for allowedType in allowedTypes[1:]:
                allowedTypeStr += ", " + allowedType
            s += f"<dt><tt>{type_constraint.type_param_str}</tt> : {allowedTypeStr}</dt>\n"
            s += f"<dd>{type_constraint.description}</dd>\n"
        s += "</dl>\n"

    # Function Body
    # TODO: this should be refactored to show the function body graph's picture (DAG).
    # if schema.has_function or schema.has_context_dependent_function:  # type: ignore
    #    s += '\n#### Function\n'
    #    s += '\nThe Function can be represented as a function.\n'

    return s


def support_level_str(level: OpSchema.SupportType) -> str:
    return (
        "<sub>experimental</sub> " if level == OpSchema.SupportType.EXPERIMENTAL else ""
    )


class Args(NamedTuple):
    output: str
    changelog: str


def main(args: Args) -> None:
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    docs_dir = os.path.join(base_dir, "docs")

    with open(
        os.path.join(docs_dir, args.changelog), "w", newline="", encoding="utf-8"
    ) as fout:
        fout.write("<!--- SPDX-License-Identifier: Apache-2.0 -->\n")
        fout.write("## Operator Changelog\n")
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n"
            "\n"
            "For an operator input/output's differentiability, it can be differentiable,\n"
            "            non-differentiable, or undefined. If a variable's differentiability\n"
            "            is not specified, that variable has undefined differentiability.\n"
        )

        # domain -> version -> [schema]
        dv_index: Dict[str, Dict[int, List[OpSchema]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for schema in defs.get_all_schemas_with_history():
            dv_index[schema.domain][schema.since_version].append(schema)

        fout.write("\n")

        for domain, versionmap in sorted(dv_index.items()):
            if not should_render_domain(domain, args.output):
                continue

            s = f"# {display_domain_short(domain)}\n"

            for version, unsorted_schemas in sorted(versionmap.items()):
                s += f"## Version {version} of {display_domain(domain)}\n"
                for schema in sorted(unsorted_schemas, key=lambda s: s.name):
                    name_with_ver = f"{format_name_with_domain(domain, schema.name)}-{schema.since_version}"
                    s += (
                        '### <a name="{}"></a>**{}**'
                        + (" (deprecated)" if schema.deprecated else "")
                        + "</a>\n"
                    ).format(name_with_ver, name_with_ver)
                    s += display_schema(schema, [schema], args.changelog)
                    s += "\n"

            fout.write(s)

    with open(
        os.path.join(docs_dir, args.output), "w", newline="", encoding="utf-8"
    ) as fout:
        fout.write("<!--- SPDX-License-Identifier: Apache-2.0 -->\n")
        fout.write("## Operator Schemas\n")
        fout.write(
            "*This file is automatically generated from the\n"
            "            [def files](/onnx/defs) via [this script](/onnx/defs/gen_doc.py).\n"
            "            Do not modify directly and instead edit operator definitions.*\n"
            "\n"
            "For an operator input/output's differentiability, it can be differentiable,\n"
            "            non-differentiable, or undefined. If a variable's differentiability\n"
            "            is not specified, that variable has undefined differentiability.\n"
        )

        # domain -> support level -> name -> [schema]
        index: Dict[str, Dict[int, Dict[str, List[OpSchema]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for schema in defs.get_all_schemas_with_history():
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

        fout.write("\n")

        # Preprocess the Operator Schemas
        # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
        operator_schemas: List[
            Tuple[str, List[Tuple[int, List[Tuple[str, OpSchema, List[OpSchema]]]]]]
        ] = []
        existing_ops: Set[str] = set()
        for domain, _supportmap in sorted(index.items()):
            if not should_render_domain(domain, args.output):
                continue

            processed_supportmap = []
            for _support, _namemap in sorted(_supportmap.items()):
                processed_namemap = []
                for n, unsorted_versions in sorted(_namemap.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    schema = versions[-1]
                    if schema.name in existing_ops:
                        continue
                    existing_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))
                processed_supportmap.append((_support, processed_namemap))
            operator_schemas.append((domain, processed_supportmap))

        # Table of contents
        for domain, supportmap in operator_schemas:
            s = f"### {display_domain_short(domain)}\n"
            fout.write(s)

            fout.write("|**Operator**|**Since version**||\n")
            fout.write("|-|-|-|\n")

            function_ops = []
            for _, namemap in supportmap:
                for n, schema, versions in namemap:
                    if schema.has_function or schema.has_context_dependent_function:  # type: ignore
                        function_versions = schema.all_function_opset_versions  # type: ignore
                        function_ops.append((n, schema, versions, function_versions))
                        continue
                    s = '|{}<a href="#{}">{}</a>{}|{}|\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n),
                        " (deprecated)" if schema.deprecated else "",
                        format_versions(versions, args.changelog),
                    )
                    fout.write(s)
            if function_ops:
                fout.write("|**Function**|**Since version**|**Function version**|\n")
                for n, schema, versions, function_versions in function_ops:
                    s = '|{}<a href="#{}">{}</a>|{}|{}|\n'.format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, n),
                        format_name_with_domain(domain, n),
                        format_versions(versions, args.changelog),
                        format_function_versions(function_versions),
                    )
                    fout.write(s)

            fout.write("\n")

        fout.write("\n")

        for domain, supportmap in operator_schemas:
            s = f"## {display_domain_short(domain)}\n"
            fout.write(s)

            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    # op_type
                    s = (
                        '### {}<a name="{}"></a><a name="{}">**{}**'
                        + (" (deprecated)" if schema.deprecated else "")
                        + "</a>\n"
                    ).format(
                        support_level_str(schema.support_level),
                        format_name_with_domain(domain, op_type),
                        format_name_with_domain(domain, op_type.lower()),
                        format_name_with_domain(domain, op_type),
                    )

                    s += display_schema(schema, versions, args.changelog)

                    s += "\n\n"

                    if op_type in SNIPPETS:
                        s += "#### Examples\n\n"
                        for summary, code in sorted(SNIPPETS[op_type]):
                            s += "<details>\n"
                            s += f"<summary>{summary}</summary>\n\n"
                            s += f"```python\n{code}\n```\n\n"
                            s += "</details>\n"
                            s += "\n\n"
                    if op_type.lower() in SAMPLE_IMPLEMENTATIONS:
                        s += "#### Sample Implementation\n\n"
                        s += "<details>\n"
                        s += f"<summary>{op_type}</summary>\n\n"
                        s += f"```python\n{SAMPLE_IMPLEMENTATIONS[op_type.lower()]}\n```\n\n"
                        s += "</details>\n"
                        s += "\n\n"

                    fout.write(s)


if __name__ == "__main__":
    if ONNX_ML:
        main(
            Args(
                "Operators-ml.md",
                "Changelog-ml.md",
            )
        )
    main(
        Args(
            "Operators.md",
            "Changelog.md",
        )
    )
