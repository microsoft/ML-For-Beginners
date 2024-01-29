# SPDX-License-Identifier: Apache-2.0

import textwrap
import onnx
import onnx.defs  # noqa
from onnx.defs import OpSchema


def _get_doc_template():
    try:
        from jinja2 import Template
    except ImportError:

        class Template:
            def __init__(self, *args):
                pass

            def render(self, **context):
                schemas = context["schemas"]
                rows = []
                for sch in schemas:
                    doc = sch.doc or ""
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name), "", doc, ""])
                return "\n".join(rows)

    return Template(
        textwrap.dedent(
            """
        {% for sch in schemas %}

        {{format_name_with_domain(sch)}}
        {{'=' * len(format_name_with_domain(sch))}}

        **Version**

        *Onnx name:* `{{sch.name}} <{{build_doc_url(sch)}}{{sch.name}}>`_

        {% if sch.support_level == OpSchema.SupportType.EXPERIMENTAL %}
        No versioning maintained for experimental ops.
        {% else %}
        This version of the operator has been {% if
        sch.deprecated %}deprecated{% else %}available{% endif %} since
        version {{sch.since_version}}{% if
        sch.domain %} of domain {{sch.domain}}{% endif %}.
        {% if len(sch.versions) > 1 %}
        Other versions of this operator:
        {% for v in sch.version[:-1] %} {{v}} {% endfor %}
        {% endif %}
        {% endif %}

        **Summary**

        {{process_documentation(sch.doc)}}

        {% if sch.attributes %}
        **Attributes**

        {% for _, attr in sorted(sch.attributes.items()) %}* *{{attr.name}}*{%
          if attr.required %} (required){% endif %}: {{attr.description}} {%
          if attr.default_value %}Default value is
          ``{{str(attr.default_value).replace('\\n', ' ').strip()}}``{%
          endif %}
        {% endfor %}
        {% endif %}

        {% if sch.inputs %}
        **Inputs**

        {% if sch.min_input != sch.max_input %}Between {{sch.min_input
        }} and {{sch.max_input}} inputs.
        {% endif %}
        {% for ii, inp in enumerate(sch.inputs) %}
        * *{{getname(inp, ii)}}*{{format_option(inp)}}{{get_type_str(inp)}}: {{
        inp.description}}{% endfor %}
        {% endif %}

        {% if sch.outputs %}
        **Outputs**

        {% if sch.min_output != sch.max_output %}Between {{sch.min_output
        }} and {{sch.max_output}} outputs.
        {% endif %}
        {% for ii, out in enumerate(sch.outputs) %}
        * *{{getname(out, ii)}}*{{format_option(out)}}{{get_type_str(out)}}: {{
        out.description}}{% endfor %}
        {% endif %}

        {% if sch.type_constraints %}
        **Type Constraints**

        {% for ii, type_constraint in enumerate(sch.type_constraints)
        %}* {{getconstraint(type_constraint, ii)}}: {{
        type_constraint.description}}
        {% endfor %}
        {% endif %}

        {% endfor %}
    """
        )
    )


_template_operator = _get_doc_template()


def get_domain_list():
    """
    Returns the list of available domains.
    """
    return list(
        sorted(set(map(lambda s: s.domain, onnx.defs.get_all_schemas_with_history())))
    )


def get_rst_doc(op_name=None):
    """
    Returns a documentation in RST format
    for all :class:`OnnxOperator`.

    :param op_name: operator name of None for all
    :return: string

    The function relies on module *jinja2* or replaces it
    with a simple rendering if not present.
    """
    if op_name is None:
        schemas = onnx.defs.get_all_schemas_with_history()
    elif isinstance(op_name, str):
        schemas = [
            schema
            for schema in onnx.defs.get_all_schemas_with_history()
            if schema.name == op_name
        ]
        if len(schemas) > 1:
            raise RuntimeError(
                "Multiple operators have the same name '{}'.".format(op_name)
            )
    elif not isinstance(op_name, list):
        schemas = [op_name]
    if len(schemas) == 0:
        raise ValueError("Unable to find any operator with name '{}'.".format(op_name))

    # from onnx.backend.sample.ops import collect_sample_implementations
    # from onnx.backend.test.case import collect_snippets
    # SNIPPETS = collect_snippets()
    # SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
    def format_name_with_domain(sch):
        if sch.domain:
            return "{} ({})".format(sch.name, sch.domain)
        return sch.name

    def get_is_homogeneous(obj):
        try:
            return obj.is_homogeneous
        except AttributeError:
            try:
                return obj.isHomogeneous
            except AttributeError:
                return False

    def format_option(obj):
        opts = []
        if OpSchema.FormalParameterOption.Optional == obj.option:
            opts.append("optional")
        elif OpSchema.FormalParameterOption.Variadic == obj.option:
            opts.append("variadic")
        if get_is_homogeneous(obj):
            opts.append("heterogeneous")
        if opts:
            return " (%s)" % ", ".join(opts)
        return ""

    def getconstraint(const, ii):
        if const.type_param_str:
            name = const.type_param_str
        else:
            name = str(ii)
        if const.allowed_type_strs:
            name += " " + ", ".join(const.allowed_type_strs)
        return name

    def getname(obj, i):
        name = obj.name
        if len(name) == 0:
            return str(i)
        return name

    def process_documentation(doc):
        if doc is None:
            doc = ""
        doc = textwrap.dedent(doc)
        main_docs_url = "https://github.com/onnx/onnx/blob/master/"
        rep = {
            "[the doc](IR.md)": "`ONNX <{0}docs/IR.md>`_",
            "[the doc](Broadcasting.md)": (
                "`Broadcasting in ONNX <{0}docs/Broadcasting.md>`_"
            ),
            "<dl>": "",
            "</dl>": "",
            "<dt>": "* ",
            "<dd>": "  ",
            "</dt>": "",
            "</dd>": "",
            "<tt>": "``",
            "</tt>": "``",
            "<br>": "\n",
        }
        for k, v in rep.items():
            doc = doc.replace(k, v.format(main_docs_url))
        move = 0
        lines = []
        for line in doc.split("\n"):
            if line.startswith("```"):
                if move > 0:
                    move -= 4
                    lines.append("\n")
                else:
                    lines.append("::\n")
                    move += 4
            elif move > 0:
                lines.append(" " * move + line)
            else:
                lines.append(line)
        return "\n".join(lines)

    def build_doc_url(sch):
        doc_url = "https://github.com/onnx/onnx/blob/master/docs/Operators"
        if "ml" in sch.domain:
            doc_url += "-ml"
        doc_url += ".md"
        doc_url += "#"
        if sch.domain not in (None, "", "ai.onnx"):
            doc_url += sch.domain + "."
        return doc_url

    def get_type_str(inou):
        try:
            return inou.type_str
        except AttributeError:
            return inou.typeStr

    fnwd = format_name_with_domain
    tmpl = _template_operator
    docs = tmpl.render(
        schemas=schemas,
        OpSchema=OpSchema,
        len=len,
        getattr=getattr,
        sorted=sorted,
        format_option=format_option,
        getconstraint=getconstraint,
        getname=getname,
        enumerate=enumerate,
        format_name_with_domain=fnwd,
        process_documentation=process_documentation,
        build_doc_url=build_doc_url,
        str=str,
        get_type_str=get_type_str,
    )
    return docs


def _get_doc_template_sklearn():
    try:
        from jinja2 import Template
    except ImportError:

        class Template:
            def __init__(self, *args):
                pass

            def render(self, **context):
                schemas = context["schemas"]
                rows = []
                for sch in schemas:
                    doc = sch.doc or ""
                    name = sch.name
                    if name is None:
                        raise RuntimeError("An operator must have a name.")
                    rows.extend([name, "=" * len(name), "", doc, ""])
                return "\n".join(rows)

    return Template(
        textwrap.dedent(
            """
        {% for cl in classes %}

        .. _l-sklops-{{cl.__name__}}:

        {{cl.__name__}}
        {{'=' * len(cl.__name__)}}

        Corresponding :class:`OnnxSubGraphOperatorMixin
        <skl2onnx.algebra.onnx_subgraph_operator_mixin.
        OnnxSubGraphOperatorMixin>` for model
        **{{cl.operator_name}}**.

        * Shape calculator: *{{cl._fct_shape_calc.__name__}}*
        * Converter: *{{cl._fct_converter.__name__}}*

        {{format_doc(cl)}}

        {% endfor %}
    """
        )
    )


_template_operator_sklearn = _get_doc_template_sklearn()


def get_rst_doc_sklearn():
    """
    Returns a documentation in RST format
    for all :class:`OnnxSubGraphOperatorMixin`.

    :param op_name: operator name of None for all
    :return: string

    The function relies on module *jinja2* or replaces it
    with a simple rendering if not present.
    """

    def format_doc(cl):
        return "\n".join(cl.__doc__.split("\n")[1:])

    from .sklearn_ops import dynamic_class_creation_sklearn

    classes = dynamic_class_creation_sklearn()
    tmpl = _template_operator_sklearn
    values = [(k, v) for k, v in sorted(classes.items())]
    values = [_[1] for _ in values]
    docs = tmpl.render(len=len, classes=values, format_doc=format_doc)
    return docs
