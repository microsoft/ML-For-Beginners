from ast import Module
from ast import stmt
from typing import List as ListType

import ast
import inspect
from ast import PyCF_ONLY_AST, PyCF_ALLOW_TOP_LEVEL_AWAIT
import types
import os
from contextlib import contextmanager
PyCF_DONT_IMPLY_DEDENT = 0x200  # Matches pythonrun.h

_assign_nodes = (ast.AugAssign, ast.AnnAssign, ast.Assign)
_single_targets_nodes = (ast.AugAssign, ast.AnnAssign)

user_module = types.ModuleType("__main__",
                               doc="Automatically created module for IPython interactive environment")

stored = []


def tracefunc(frame, event, arg):
    if '_debugger_case_scoped_stepping_target' not in frame.f_code.co_filename:
        return None
    stored.append(frame)
    print('\n---')
    print(event, id(frame), os.path.basename(frame.f_code.co_filename), frame.f_lineno, arg, frame.f_code.co_name)
    assert frame.f_back.f_code.co_name == 'run_code'
    return None


@contextmanager
def tracing_info():
    import sys
    sys.settrace(tracefunc)
    try:
        yield
    finally:
        sys.settrace(None)


# Note: this is roughly what IPython itself does at:
# https://github.com/ipython/ipython/blob/master/IPython/core/interactiveshell.py
class Runner:

    async def run_ast_nodes(
        self,
        nodelist: ListType[stmt],
        cell_name: str,
        interactivity="last_expr",
        compiler=compile,
    ):
        if not nodelist:
            return

        if interactivity == 'last_expr_or_assign':
            if isinstance(nodelist[-1], _assign_nodes):
                asg = nodelist[-1]
                if isinstance(asg, ast.Assign) and len(asg.targets) == 1:
                    target = asg.targets[0]
                elif isinstance(asg, _single_targets_nodes):
                    target = asg.target
                else:
                    target = None
                if isinstance(target, ast.Name):
                    nnode = ast.Expr(ast.Name(target.id, ast.Load()))
                    ast.fix_missing_locations(nnode)
                    nodelist.append(nnode)
            interactivity = 'last_expr'

        _async = False
        if interactivity == 'last_expr':
            if isinstance(nodelist[-1], ast.Expr):
                interactivity = "last"
            else:
                interactivity = "none"

        if interactivity == 'none':
            to_run_exec, to_run_interactive = nodelist, []
        elif interactivity == 'last':
            to_run_exec, to_run_interactive = nodelist[:-1], nodelist[-1:]
        elif interactivity == 'all':
            to_run_exec, to_run_interactive = [], nodelist
        else:
            raise ValueError("Interactivity was %r" % interactivity)

        def compare(code):
            is_async = inspect.CO_COROUTINE & code.co_flags == inspect.CO_COROUTINE
            return is_async

        # Refactor that to just change the mod constructor.
        to_run = []
        for node in to_run_exec:
            to_run.append((node, "exec"))

        for node in to_run_interactive:
            to_run.append((node, "single"))

        for node, mode in to_run:
            if mode == "exec":
                mod = Module([node], [])
            elif mode == "single":
                mod = ast.Interactive([node])
            code = compiler(mod, cell_name, mode, PyCF_DONT_IMPLY_DEDENT |
                            PyCF_ALLOW_TOP_LEVEL_AWAIT)
            asy = compare(code)
            if await self.run_code(code, async_=asy):
                return True

    async def run_code(self, code_obj, *, async_=False):
        if async_:
            await eval(code_obj, self.user_global_ns, self.user_ns)
        else:
            exec(code_obj, self.user_global_ns, self.user_ns)

    @property
    def user_global_ns(self):
        return user_module.__dict__

    @property
    def user_ns(self):
        return user_module.__dict__


async def main():
    SCOPED_STEPPING_TARGET = os.getenv('SCOPED_STEPPING_TARGET', '_debugger_case_scoped_stepping_target.py')
    filename = os.path.join(os.path.dirname(__file__), SCOPED_STEPPING_TARGET)
    assert os.path.exists(filename), '%s does not exist.' % (filename,)
    with open(filename, 'r') as stream:
        source = stream.read()
    code_ast = compile(
        source,
        filename,
        'exec',
        PyCF_DONT_IMPLY_DEDENT | PyCF_ONLY_AST | PyCF_ALLOW_TOP_LEVEL_AWAIT,
        1)

    runner = Runner()
    await runner.run_ast_nodes(code_ast.body, filename)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    print('TEST SUCEEDED!')
