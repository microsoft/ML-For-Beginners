'''
Run this module to regenerate the `pydevd_schema.py` file.

Note that it'll generate it based on the current debugProtocol.json. Erase it and rerun
to download the latest version.
'''


def is_variable_to_translate(cls_name, var_name):
    if var_name in ('variablesReference', 'frameId', 'threadId'):
        return True

    if cls_name == 'StackFrame' and var_name == 'id':
        # It's frameId everywhere except on StackFrame.
        return True

    if cls_name == 'Thread' and var_name == 'id':
        # It's threadId everywhere except on Thread.
        return True

    return False


def _get_noqa_for_var(prop_name):
    return '  # noqa (assign to builtin)' if prop_name in ('type', 'format', 'id', 'hex', 'breakpoint', 'filter') else ''


class _OrderedSet(object):
    # Not a good ordered set (just something to be small without adding any deps)

    def __init__(self, initial_contents=None):
        self._contents = []
        self._contents_as_set = set()
        if initial_contents is not None:
            for x in initial_contents:
                self.add(x)

    def add(self, x):
        if x not in self._contents_as_set:
            self._contents_as_set.add(x)
            self._contents.append(x)

    def discard(self, x):
        if x in self._contents_as_set:
            self._contents_as_set.remove(x)
            self._contents.remove(x)

    def copy(self):
        return _OrderedSet(self._contents)

    def update(self, contents):
        for x in contents:
            self.add(x)

    def __iter__(self):
        return iter(self._contents)

    def __contains__(self, item):
        return item in self._contents_as_set

    def __len__(self):
        return len(self._contents)

    def set_repr(self):
        if len(self) == 0:
            return 'set()'

        lst = [repr(x) for x in self]
        return 'set([' + ', '.join(lst) + '])'


class Ref(object):

    def __init__(self, ref, ref_data):
        self.ref = ref
        self.ref_data = ref_data

    def __str__(self):
        return self.ref


def load_schema_data():
    import os.path
    import json

    json_file = os.path.join(os.path.dirname(__file__), 'debugProtocol.json')
    if not os.path.exists(json_file):
        import requests
        req = requests.get('https://raw.githubusercontent.com/microsoft/debug-adapter-protocol/gh-pages/debugAdapterProtocol.json')
        assert req.status_code == 200
        with open(json_file, 'wb') as stream:
            stream.write(req.content)

    with open(json_file, 'rb') as json_contents:
        json_schema_data = json.loads(json_contents.read())
    return json_schema_data


def load_custom_schema_data():
    import os.path
    import json

    json_file = os.path.join(os.path.dirname(__file__), 'debugProtocolCustom.json')

    with open(json_file, 'rb') as json_contents:
        json_schema_data = json.loads(json_contents.read())
    return json_schema_data


def create_classes_to_generate_structure(json_schema_data):
    definitions = json_schema_data['definitions']

    class_to_generatees = {}

    for name, definition in definitions.items():
        all_of = definition.get('allOf')
        description = definition.get('description')
        is_enum = definition.get('type') == 'string' and 'enum' in definition
        enum_values = None
        if is_enum:
            enum_values = definition['enum']
        properties = {}
        properties.update(definition.get('properties', {}))
        required = _OrderedSet(definition.get('required', _OrderedSet()))
        base_definitions = []

        if all_of is not None:
            for definition in all_of:
                ref = definition.get('$ref')
                if ref is not None:
                    assert ref.startswith('#/definitions/')
                    ref = ref[len('#/definitions/'):]
                    base_definitions.append(ref)
                else:
                    if not description:
                        description = definition.get('description')
                    properties.update(definition.get('properties', {}))
                    required.update(_OrderedSet(definition.get('required', _OrderedSet())))

        if isinstance(description, (list, tuple)):
            description = '\n'.join(description)

        if name == 'ModulesRequest':  # Hack to accept modules request without arguments (ptvsd: 2050).
            required.discard('arguments')
        class_to_generatees[name] = dict(
            name=name,
            properties=properties,
            base_definitions=base_definitions,
            description=description,
            required=required,
            is_enum=is_enum,
            enum_values=enum_values
        )
    return class_to_generatees


def collect_bases(curr_class, classes_to_generate, memo=None):
    ret = []
    if memo is None:
        memo = {}

    base_definitions = curr_class['base_definitions']
    for base_definition in base_definitions:
        if base_definition not in memo:
            ret.append(base_definition)
            ret.extend(collect_bases(classes_to_generate[base_definition], classes_to_generate, memo))

    return ret


def fill_properties_and_required_from_base(classes_to_generate):
    # Now, resolve properties based on refs
    for class_to_generate in classes_to_generate.values():
        dct = {}
        s = _OrderedSet()

        for base_definition in reversed(collect_bases(class_to_generate, classes_to_generate)):
            # Note: go from base to current so that the initial order of the properties has that
            # same order.
            dct.update(classes_to_generate[base_definition].get('properties', {}))
            s.update(classes_to_generate[base_definition].get('required', _OrderedSet()))

        dct.update(class_to_generate['properties'])
        class_to_generate['properties'] = dct

        s.update(class_to_generate['required'])
        class_to_generate['required'] = s

    return class_to_generate


def update_class_to_generate_description(class_to_generate):
    import textwrap
    description = class_to_generate['description']
    lines = []
    for line in description.splitlines():
        wrapped = textwrap.wrap(line.strip(), 100)
        lines.extend(wrapped)
        lines.append('')

    while lines and lines[-1] == '':
        lines = lines[:-1]

    class_to_generate['description'] = '    ' + ('\n    '.join(lines))


def update_class_to_generate_type(classes_to_generate, class_to_generate):
    properties = class_to_generate.get('properties')
    for _prop_name, prop_val in properties.items():
        prop_type = prop_val.get('type', '')
        if not prop_type:
            prop_type = prop_val.pop('$ref', '')
            if prop_type:
                assert prop_type.startswith('#/definitions/')
                prop_type = prop_type[len('#/definitions/'):]
                prop_val['type'] = Ref(prop_type, classes_to_generate[prop_type])


def update_class_to_generate_register_dec(classes_to_generate, class_to_generate):
    # Default
    class_to_generate['register_request'] = ''
    class_to_generate['register_dec'] = '@register'

    properties = class_to_generate.get('properties')
    enum_type = properties.get('type', {}).get('enum')
    command = None
    event = None
    if enum_type and len(enum_type) == 1 and next(iter(enum_type)) in ("request", "response", "event"):
        msg_type = next(iter(enum_type))
        if msg_type == 'response':
            # The actual command is typed in the request
            response_name = class_to_generate['name']
            request_name = response_name[:-len('Response')] + 'Request'
            if request_name in classes_to_generate:
                command = classes_to_generate[request_name]['properties'].get('command')
            else:
                if response_name == 'ErrorResponse':
                    command = {'enum': ['error']}
                else:
                    raise AssertionError('Unhandled: %s' % (response_name,))

        elif msg_type == 'request':
            command = properties.get('command')

        elif msg_type == 'event':
            command = properties.get('event')

        else:
            raise AssertionError('Unexpected condition.')

        if command:
            enum = command.get('enum')
            if enum and len(enum) == 1:
                class_to_generate['register_request'] = '@register_%s(%r)\n' % (msg_type, enum[0])


def extract_prop_name_and_prop(class_to_generate):
    properties = class_to_generate.get('properties')
    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))

    # Sort so that required come first
    prop_name_and_prop = list(properties.items())

    def compute_sort_key(x):
        key = x[0]
        if key in required:
            if key == 'seq':
                return 0.5  # seq when required is after the other required keys (to have a default of -1).
            return 0
        return 1

    prop_name_and_prop.sort(key=compute_sort_key)

    return prop_name_and_prop


def update_class_to_generate_to_json(class_to_generate):
    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))
    prop_name_and_prop = extract_prop_name_and_prop(class_to_generate)

    to_dict_body = ['def to_dict(self, update_ids_to_dap=False):  # noqa (update_ids_to_dap may be unused)']

    translate_prop_names = []
    for prop_name, prop in prop_name_and_prop:
        if is_variable_to_translate(class_to_generate['name'], prop_name):
            translate_prop_names.append(prop_name)

    for prop_name, prop in prop_name_and_prop:
        namespace = dict(prop_name=prop_name, noqa=_get_noqa_for_var(prop_name))
        to_dict_body.append('    %(prop_name)s = self.%(prop_name)s%(noqa)s' % namespace)

        if prop.get('type') == 'array':
            to_dict_body.append('    if %(prop_name)s and hasattr(%(prop_name)s[0], "to_dict"):' % namespace)
            to_dict_body.append('        %(prop_name)s = [x.to_dict() for x in %(prop_name)s]' % namespace)

    if translate_prop_names:
        to_dict_body.append('    if update_ids_to_dap:')
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name, noqa=_get_noqa_for_var(prop_name))
            to_dict_body.append('        if %(prop_name)s is not None:' % namespace)
            to_dict_body.append('            %(prop_name)s = self._translate_id_to_dap(%(prop_name)s)%(noqa)s' % namespace)

    if not translate_prop_names:
        update_dict_ids_from_dap_body = []
    else:
        update_dict_ids_from_dap_body = ['', '', '@classmethod', 'def update_dict_ids_from_dap(cls, dct):']
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name)
            update_dict_ids_from_dap_body.append('    if %(prop_name)r in dct:' % namespace)
            update_dict_ids_from_dap_body.append('        dct[%(prop_name)r] = cls._translate_id_from_dap(dct[%(prop_name)r])' % namespace)
        update_dict_ids_from_dap_body.append('    return dct')

    class_to_generate['update_dict_ids_from_dap'] = _indent_lines('\n'.join(update_dict_ids_from_dap_body))

    to_dict_body.append('    dct = {')
    first_not_required = False

    for prop_name, prop in prop_name_and_prop:
        use_to_dict = prop['type'].__class__ == Ref and not prop['type'].ref_data.get('is_enum', False)
        is_array = prop['type'] == 'array'
        ref_array_cls_name = ''
        if is_array:
            ref = prop['items'].get('$ref')
            if ref is not None:
                ref_array_cls_name = ref.split('/')[-1]

        namespace = dict(prop_name=prop_name, ref_array_cls_name=ref_array_cls_name)
        if prop_name in required:
            if use_to_dict:
                to_dict_body.append('        %(prop_name)r: %(prop_name)s.to_dict(update_ids_to_dap=update_ids_to_dap),' % namespace)
            else:
                if ref_array_cls_name:
                    to_dict_body.append('        %(prop_name)r: [%(ref_array_cls_name)s.update_dict_ids_to_dap(o) for o in %(prop_name)s] if (update_ids_to_dap and %(prop_name)s) else %(prop_name)s,' % namespace)
                else:
                    to_dict_body.append('        %(prop_name)r: %(prop_name)s,' % namespace)
        else:
            if not first_not_required:
                first_not_required = True
                to_dict_body.append('    }')

            to_dict_body.append('    if %(prop_name)s is not None:' % namespace)
            if use_to_dict:
                to_dict_body.append('        dct[%(prop_name)r] = %(prop_name)s.to_dict(update_ids_to_dap=update_ids_to_dap)' % namespace)
            else:
                if ref_array_cls_name:
                    to_dict_body.append('        dct[%(prop_name)r] = [%(ref_array_cls_name)s.update_dict_ids_to_dap(o) for o in %(prop_name)s] if (update_ids_to_dap and %(prop_name)s) else %(prop_name)s' % namespace)
                else:
                    to_dict_body.append('        dct[%(prop_name)r] = %(prop_name)s' % namespace)

    if not first_not_required:
        first_not_required = True
        to_dict_body.append('    }')

    to_dict_body.append('    dct.update(self.kwargs)')
    to_dict_body.append('    return dct')

    class_to_generate['to_dict'] = _indent_lines('\n'.join(to_dict_body))

    if not translate_prop_names:
        update_dict_ids_to_dap_body = []
    else:
        update_dict_ids_to_dap_body = ['', '', '@classmethod', 'def update_dict_ids_to_dap(cls, dct):']
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name)
            update_dict_ids_to_dap_body.append('    if %(prop_name)r in dct:' % namespace)
            update_dict_ids_to_dap_body.append('        dct[%(prop_name)r] = cls._translate_id_to_dap(dct[%(prop_name)r])' % namespace)
        update_dict_ids_to_dap_body.append('    return dct')

    class_to_generate['update_dict_ids_to_dap'] = _indent_lines('\n'.join(update_dict_ids_to_dap_body))


def update_class_to_generate_init(class_to_generate):
    args = []
    init_body = []
    docstring = []

    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))
    prop_name_and_prop = extract_prop_name_and_prop(class_to_generate)

    translate_prop_names = []
    for prop_name, prop in prop_name_and_prop:
        if is_variable_to_translate(class_to_generate['name'], prop_name):
            translate_prop_names.append(prop_name)

        enum = prop.get('enum')
        if enum and len(enum) == 1:
            init_body.append('    self.%(prop_name)s = %(enum)r' % dict(prop_name=prop_name, enum=next(iter(enum))))
        else:
            if prop_name in required:
                if prop_name == 'seq':
                    args.append(prop_name + '=-1')
                else:
                    args.append(prop_name)
            else:
                args.append(prop_name + '=None')

            if prop['type'].__class__ == Ref:
                ref = prop['type']
                ref_data = ref.ref_data
                if ref_data.get('is_enum', False):
                    init_body.append('    if %s is not None:' % (prop_name,))
                    init_body.append('        assert %s in %s.VALID_VALUES' % (prop_name, str(ref)))
                    init_body.append('    self.%(prop_name)s = %(prop_name)s' % dict(
                        prop_name=prop_name))
                else:
                    namespace = dict(
                        prop_name=prop_name,
                        ref_name=str(ref)
                    )
                    init_body.append('    if %(prop_name)s is None:' % namespace)
                    init_body.append('        self.%(prop_name)s = %(ref_name)s()' % namespace)
                    init_body.append('    else:')
                    init_body.append('        self.%(prop_name)s = %(ref_name)s(update_ids_from_dap=update_ids_from_dap, **%(prop_name)s) if %(prop_name)s.__class__ !=  %(ref_name)s else %(prop_name)s' % namespace
                    )

            else:
                init_body.append('    self.%(prop_name)s = %(prop_name)s' % dict(prop_name=prop_name))

                if prop['type'] == 'array':
                    ref = prop['items'].get('$ref')
                    if ref is not None:
                        ref_array_cls_name = ref.split('/')[-1]
                        init_body.append('    if update_ids_from_dap and self.%(prop_name)s:' % dict(prop_name=prop_name))
                        init_body.append('        for o in self.%(prop_name)s:' % dict(prop_name=prop_name))
                        init_body.append('            %(ref_array_cls_name)s.update_dict_ids_from_dap(o)' % dict(ref_array_cls_name=ref_array_cls_name))

        prop_type = prop['type']
        prop_description = prop.get('description', '')

        if isinstance(prop_description, (list, tuple)):
            prop_description = '\n    '.join(prop_description)

        docstring.append(':param %(prop_type)s %(prop_name)s: %(prop_description)s' % dict(
            prop_type=prop_type, prop_name=prop_name, prop_description=prop_description))

    if translate_prop_names:
        init_body.append('    if update_ids_from_dap:')
        for prop_name in translate_prop_names:
            init_body.append('        self.%(prop_name)s = self._translate_id_from_dap(self.%(prop_name)s)' % dict(prop_name=prop_name))

    docstring = _indent_lines('\n'.join(docstring))
    init_body = '\n'.join(init_body)

    # Actually bundle the whole __init__ from the parts.
    args = ', '.join(args)
    if args:
        args = ', ' + args

    # Note: added kwargs because some messages are expected to be extended by the user (so, we'll actually
    # make all extendable so that we don't have to worry about which ones -- we loose a little on typing,
    # but may be better than doing a allow list based on something only pointed out in the documentation).
    class_to_generate['init'] = '''def __init__(self%(args)s, update_ids_from_dap=False, **kwargs):  # noqa (update_ids_from_dap may be unused)
    """
%(docstring)s
    """
%(init_body)s
    self.kwargs = kwargs
''' % dict(args=args, init_body=init_body, docstring=docstring)

    class_to_generate['init'] = _indent_lines(class_to_generate['init'])


def update_class_to_generate_props(class_to_generate):
    import json

    def default(o):
        if isinstance(o, Ref):
            return o.ref
        raise AssertionError('Unhandled: %s' % (o,))

    properties = class_to_generate['properties']
    class_to_generate['props'] = '    __props__ = %s' % _indent_lines(
        json.dumps(properties, indent=4, default=default)).strip()


def update_class_to_generate_refs(class_to_generate):
    properties = class_to_generate['properties']
    class_to_generate['refs'] = '    __refs__ = %s' % _OrderedSet(
        key for (key, val) in properties.items() if val['type'].__class__ == Ref).set_repr()


def update_class_to_generate_enums(class_to_generate):
    class_to_generate['enums'] = ''
    if class_to_generate.get('is_enum', False):
        enums = ''
        for enum in class_to_generate['enum_values']:
            enums += '    %s = %r\n' % (enum.upper(), enum)
        enums += '\n'
        enums += '    VALID_VALUES = %s\n\n' % _OrderedSet(class_to_generate['enum_values']).set_repr()
        class_to_generate['enums'] = enums


def update_class_to_generate_objects(classes_to_generate, class_to_generate):
    properties = class_to_generate['properties']
    for key, val in properties.items():
        if 'type' not in val:
            val['type'] = 'TypeNA'
            continue

        if val['type'] == 'object':
            create_new = val.copy()
            create_new.update({
                'name': '%s%s' % (class_to_generate['name'], key.title()),
                'description': '    "%s" of %s' % (key, class_to_generate['name'])
            })
            if 'properties' not in create_new:
                create_new['properties'] = {}

            assert create_new['name'] not in classes_to_generate
            classes_to_generate[create_new['name']] = create_new

            update_class_to_generate_type(classes_to_generate, create_new)
            update_class_to_generate_props(create_new)

            # Update nested object types
            update_class_to_generate_objects(classes_to_generate, create_new)

            val['type'] = Ref(create_new['name'], classes_to_generate[create_new['name']])
            val.pop('properties', None)


def gen_debugger_protocol():
    import os.path
    import sys

    if sys.version_info[:2] < (3, 6):
        raise AssertionError('Must be run with Python 3.6 onwards (to keep dict order).')

    classes_to_generate = create_classes_to_generate_structure(load_schema_data())
    classes_to_generate.update(create_classes_to_generate_structure(load_custom_schema_data()))

    class_to_generate = fill_properties_and_required_from_base(classes_to_generate)

    for class_to_generate in list(classes_to_generate.values()):
        update_class_to_generate_description(class_to_generate)
        update_class_to_generate_type(classes_to_generate, class_to_generate)
        update_class_to_generate_props(class_to_generate)
        update_class_to_generate_objects(classes_to_generate, class_to_generate)

    for class_to_generate in classes_to_generate.values():
        update_class_to_generate_refs(class_to_generate)
        update_class_to_generate_init(class_to_generate)
        update_class_to_generate_enums(class_to_generate)
        update_class_to_generate_to_json(class_to_generate)
        update_class_to_generate_register_dec(classes_to_generate, class_to_generate)

    class_template = '''
%(register_request)s%(register_dec)s
class %(name)s(BaseSchema):
    """
%(description)s

    Note: automatically generated code. Do not edit manually.
    """

%(enums)s%(props)s
%(refs)s

    __slots__ = list(__props__.keys()) + ['kwargs']

%(init)s%(update_dict_ids_from_dap)s

%(to_dict)s%(update_dict_ids_to_dap)s
'''

    contents = []
    contents.append('# coding: utf-8')
    contents.append('# Automatically generated code.')
    contents.append('# Do not edit manually.')
    contents.append('# Generated by running: %s' % os.path.basename(__file__))
    contents.append('from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event')
    contents.append('')
    for class_to_generate in classes_to_generate.values():
        contents.append(class_template % class_to_generate)

    parent_dir = os.path.dirname(__file__)
    schema = os.path.join(parent_dir, 'pydevd_schema.py')
    with open(schema, 'w', encoding='utf-8') as stream:
        stream.write('\n'.join(contents))


def _indent_lines(lines, indent='    '):
    out_lines = []
    for line in lines.splitlines(keepends=True):
        out_lines.append(indent + line)

    return ''.join(out_lines)


if __name__ == '__main__':

    gen_debugger_protocol()
