'''
Note: run test with manage.py test my_app

This is mostly for experimenting.

The actual code used is mostly a copy of this that lives in `django_debug.py`.
'''

from django.test import SimpleTestCase


def collect_lines_for_django_template(template_contents):
    from django import template
    t = template.Template(template_contents)
    return _collect_valid_lines_in_django_template(t)


def _collect_valid_lines_in_django_template(template):
    lines = set()
    for node in _iternodes(template.nodelist):
        lineno = _get_lineno(node)
        if lineno is not None:
            lines.add(lineno)
    return lines


def _get_lineno(node):
    if hasattr(node, 'token') and hasattr(node.token, 'lineno'):
        return node.token.lineno
    return None


def _iternodes(nodelist):
    for node in nodelist:
        yield node

        try:
            children = node.child_nodelists
        except:
            pass
        else:
            for attr in children:
                nodelist = getattr(node, attr, None)
                if nodelist:
                    # i.e.: yield from _iternodes(nodelist)
                    for node in _iternodes(nodelist):
                        yield node


class MyTest(SimpleTestCase):

    def test_something(self):
        template_contents = '''{% if entries %}
    <ul>
    {% for entry in entries %}
        {% for entry in entries2 %}
            <li>
                {{ entry.key }}
                :
                {{ entry.val }}
            </li>
        {% endfor %}
    {% endfor %}
    </ul>
{% else %}
    <p>No entries are available.</p>
{% endif %}'''

        self.assertEqual(
            collect_lines_for_django_template(template_contents),
            {1, 3, 4, 6, 8, 10, 11, 13}
        )
