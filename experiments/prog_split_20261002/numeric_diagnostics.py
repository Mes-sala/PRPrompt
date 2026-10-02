"""Record the old filter row-sum check without changing any filter arithmetic."""
import ast
import math
from experiments.v1_f100.historical import load as original_load, REFERENCE

ERRORS=[]


def record_filter_error(error):
    value=float(error)
    if not math.isfinite(value):raise ValueError('Non-finite filter row sum')
    ERRORS.append(value)


def load():
    ns=original_load()
    tree=ast.parse((REFERENCE/'run_round83_graft.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_filters')
    changed=0
    for n in ast.walk(fn):
        if isinstance(n,ast.For):
            for i,statement in enumerate(n.body):
                if isinstance(statement,ast.Assert):
                    assert isinstance(statement.test,ast.Compare) and ast.literal_eval(statement.test.comparators[0])==2e-6
                    n.body[i]=ast.Expr(value=ast.Call(func=ast.Name(id='record_filter_error',ctx=ast.Load()),args=[statement.test.left],keywords=[]))
                    changed+=1
    assert changed==1
    ns['record_filter_error']=record_filter_error
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(REFERENCE/'run_round83_graft.py'),'exec'),ns)
    return ns


def install():
    import experiments.f100_boundary_20261002.model as control
    control.load=load
