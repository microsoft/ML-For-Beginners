"""
This module contains utility function and classes to inject simple ast
transformations based on code strings into IPython. While it is already possible
with ast-transformers it is not easy to directly manipulate ast.


IPython has pre-code and post-code hooks, but are ran from within the IPython
machinery so may be inappropriate, for example for performance mesurement.

This module give you tools to simplify this, and expose 2 classes:

- `ReplaceCodeTransformer` which is a simple ast transformer based on code
  template,

and for advance case:

- `Mangler` which is a simple ast transformer that mangle names in the ast.


Example, let's try to make a simple version of the ``timeit`` magic, that run a
code snippet 10 times and print the average time taken.

Basically we want to run :

.. code-block:: python

    from time import perf_counter
    now = perf_counter()
    for i in range(10):
        __code__ # our code
    print(f"Time taken: {(perf_counter() - now)/10}")
    __ret__ # the result of the last statement

Where ``__code__`` is the code snippet we want to run, and ``__ret__`` is the
result, so that if we for example run `dataframe.head()` IPython still display
the head of dataframe instead of nothing.

Here is a complete example of a file `timit2.py` that define such a magic:

.. code-block:: python

    from IPython.core.magic import (
        Magics,
        magics_class,
        line_cell_magic,
    )
    from IPython.core.magics.ast_mod import ReplaceCodeTransformer
    from textwrap import dedent
    import ast

    template = template = dedent('''
        from time import perf_counter
        now = perf_counter()
        for i in range(10):
            __code__
        print(f"Time taken: {(perf_counter() - now)/10}")
        __ret__
    '''
    )


    @magics_class
    class AstM(Magics):
        @line_cell_magic
        def t2(self, line, cell):
            transformer = ReplaceCodeTransformer.from_string(template)
            transformer.debug = True
            transformer.mangler.debug = True
            new_code = transformer.visit(ast.parse(cell))
            return exec(compile(new_code, "<ast>", "exec"))


    def load_ipython_extension(ip):
        ip.register_magics(AstM)



.. code-block:: python

    In [1]: %load_ext timit2

    In [2]: %%t2
       ...: import time
       ...: time.sleep(0.05)
       ...:
       ...:
    Time taken: 0.05435649999999441


If you wish to ran all the code enter in IPython in an ast transformer, you can
do so as well:

.. code-block:: python

    In [1]: from IPython.core.magics.ast_mod import ReplaceCodeTransformer
       ...:
       ...: template = '''
       ...: from time import perf_counter
       ...: now = perf_counter()
       ...: __code__
       ...: print(f"Code ran in {perf_counter()-now}")
       ...: __ret__'''
       ...:
       ...: get_ipython().ast_transformers.append(ReplaceCodeTransformer.from_string(template))

    In [2]: 1+1
    Code ran in 3.40410006174352e-05
    Out[2]: 2



Hygiene and Mangling
--------------------

The ast transformer above is not hygienic, it may not work if the user code use
the same variable names as the ones used in the template. For example.

To help with this by default the `ReplaceCodeTransformer` will mangle all names
staring with 3 underscores. This is a simple heuristic that should work in most
case, but can be cumbersome in some case. We provide a `Mangler` class that can
be overridden to change the mangling heuristic, or simply use the `mangle_all`
utility function. It will _try_ to mangle all names (except `__ret__` and
`__code__`), but this include builtins (``print``, ``range``, ``type``) and
replace those by invalid identifiers py prepending ``mangle-``:
``mangle-print``, ``mangle-range``, ``mangle-type`` etc. This is not a problem
as currently Python AST support invalid identifiers, but it may not be the case
in the future.

You can set `ReplaceCodeTransformer.debug=True` and
`ReplaceCodeTransformer.mangler.debug=True` to see the code after mangling and
transforming:

.. code-block:: python


    In [1]: from IPython.core.magics.ast_mod import ReplaceCodeTransformer, mangle_all
       ...:
       ...: template = '''
       ...: from builtins import type, print
       ...: from time import perf_counter
       ...: now = perf_counter()
       ...: __code__
       ...: print(f"Code ran in {perf_counter()-now}")
       ...: __ret__'''
       ...:
       ...: transformer = ReplaceCodeTransformer.from_string(template, mangling_predicate=mangle_all)


    In [2]: transformer.debug = True
       ...: transformer.mangler.debug = True
       ...: get_ipython().ast_transformers.append(transformer)

    In [3]: 1+1
    Mangling Alias mangle-type
    Mangling Alias mangle-print
    Mangling Alias mangle-perf_counter
    Mangling now
    Mangling perf_counter
    Not mangling __code__
    Mangling print
    Mangling perf_counter
    Mangling now
    Not mangling __ret__
    ---- Transformed code ----
    from builtins import type as mangle-type, print as mangle-print
    from time import perf_counter as mangle-perf_counter
    mangle-now = mangle-perf_counter()
    ret-tmp = 1 + 1
    mangle-print(f'Code ran in {mangle-perf_counter() - mangle-now}')
    ret-tmp
    ---- ---------------- ----
    Code ran in 0.00013654199938173406
    Out[3]: 2


"""

__skip_doctest__ = True


from ast import NodeTransformer, Store, Load, Name, Expr, Assign, Module
import ast
import copy

from typing import Dict, Optional


mangle_all = lambda name: False if name in ("__ret__", "__code__") else True


class Mangler(NodeTransformer):
    """
    Mangle given names in and ast tree to make sure they do not conflict with
    user code.
    """

    enabled: bool = True
    debug: bool = False

    def log(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __init__(self, predicate=None):
        if predicate is None:
            predicate = lambda name: name.startswith("___")
        self.predicate = predicate

    def visit_Name(self, node):
        if self.predicate(node.id):
            self.log("Mangling", node.id)
            # Once in the ast we do not need
            # names to be valid identifiers.
            node.id = "mangle-" + node.id
        else:
            self.log("Not mangling", node.id)
        return node

    def visit_FunctionDef(self, node):
        if self.predicate(node.name):
            self.log("Mangling", node.name)
            node.name = "mangle-" + node.name
        else:
            self.log("Not mangling", node.name)

        for arg in node.args.args:
            if self.predicate(arg.arg):
                self.log("Mangling function arg", arg.arg)
                arg.arg = "mangle-" + arg.arg
            else:
                self.log("Not mangling function arg", arg.arg)
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        return self._visit_Import_and_ImportFrom(node)

    def visit_Import(self, node):
        return self._visit_Import_and_ImportFrom(node)

    def _visit_Import_and_ImportFrom(self, node):
        for alias in node.names:
            asname = alias.name if alias.asname is None else alias.asname
            if self.predicate(asname):
                new_name: str = "mangle-" + asname
                self.log("Mangling Alias", new_name)
                alias.asname = new_name
            else:
                self.log("Not mangling Alias", alias.asname)
        return node


class ReplaceCodeTransformer(NodeTransformer):
    enabled: bool = True
    debug: bool = False
    mangler: Mangler

    def __init__(
        self, template: Module, mapping: Optional[Dict] = None, mangling_predicate=None
    ):
        assert isinstance(mapping, (dict, type(None)))
        assert isinstance(mangling_predicate, (type(None), type(lambda: None)))
        assert isinstance(template, ast.Module)
        self.template = template
        self.mangler = Mangler(predicate=mangling_predicate)
        if mapping is None:
            mapping = {}
        self.mapping = mapping

    @classmethod
    def from_string(
        cls, template: str, mapping: Optional[Dict] = None, mangling_predicate=None
    ):
        return cls(
            ast.parse(template), mapping=mapping, mangling_predicate=mangling_predicate
        )

    def visit_Module(self, code):
        if not self.enabled:
            return code
        # if not isinstance(code, ast.Module):
        # recursively called...
        #    return generic_visit(self, code)
        last = code.body[-1]
        if isinstance(last, Expr):
            code.body.pop()
            code.body.append(Assign([Name("ret-tmp", ctx=Store())], value=last.value))
            ast.fix_missing_locations(code)
            ret = Expr(value=Name("ret-tmp", ctx=Load()))
            ret = ast.fix_missing_locations(ret)
            self.mapping["__ret__"] = ret
        else:
            self.mapping["__ret__"] = ast.parse("None").body[0]
        self.mapping["__code__"] = code.body
        tpl = ast.fix_missing_locations(self.template)

        tx = copy.deepcopy(tpl)
        tx = self.mangler.visit(tx)
        node = self.generic_visit(tx)
        node_2 = ast.fix_missing_locations(node)
        if self.debug:
            print("---- Transformed code ----")
            print(ast.unparse(node_2))
            print("---- ---------------- ----")
        return node_2

    # this does not work as the name might be in a list and one might want to extend the list.
    # def visit_Name(self, name):
    #     if name.id in self.mapping and name.id == "__ret__":
    #         print(name, "in mapping")
    #         if isinstance(name.ctx, ast.Store):
    #             return Name("tmp", ctx=Store())
    #         else:
    #             return copy.deepcopy(self.mapping[name.id])
    #     return name

    def visit_Expr(self, expr):
        if isinstance(expr.value, Name) and expr.value.id in self.mapping:
            if self.mapping[expr.value.id] is not None:
                return copy.deepcopy(self.mapping[expr.value.id])
        return self.generic_visit(expr)
