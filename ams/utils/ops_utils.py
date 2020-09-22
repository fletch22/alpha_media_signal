import sys

from IPython import get_ipython


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def initialize_notebook():
    paths_to_add = ['/home/jovyan/work']

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.append(p)
