"""Debugging helpers"""

from typing import *

__all__ = ("running_in_ipython", "set_trace", "get_signatures")


def running_in_ipython() -> bool:
    """Am I running inside an IPython session?

    Returns:
        bool: True if run within IPython session.
    """
    try:
        return __IPYTHON__
    except NameError:
        return False


def set_trace() -> Callable:
    """Sets a tracepoint in IPython or a breakpoint elsewhere

    Returns:
        Callable: The set_trace/breakpoint
    """
    if running_in_ipython():
        from IPython.core.debugger import set_trace

        return set_trace
    else:
        return breakpoint


def get_signatures(*args: Any, sep: str = " $$$ ", _print: bool = False) -> List[str]:
    """Gets signatures of all functions/classes in a module or class

    Args:
        sep (str, optional): The separator to use in signature descriptions. Defaults to " $$$ ".
        _print (bool, optional): Print instead of returning?. Defaults to False.

    Returns:
        List[str]: A list of strings with signatures, if not printing.
    """
    import inspect

    assert args
    signatures = list()

    for arg in args:
        try:
            if isinstance(arg, type(inspect)):
                if getattr(arg, "__all__", None):
                    search = arg.__all__
                else:
                    search = arg.__dir__()

                for name in search:
                    function = getattr(arg, name, None)
                    if inspect.isfunction(function):
                        signatures.append(name + sep + str(inspect.signature(function)))
            elif inspect.isclass(arg):
                for m in inspect.getmembers(arg):
                    if inspect.isroutine(m[1]):
                        name = arg.__name__ + "." + m[0]
                        signatures.append(name + sep + str(inspect.signature(m[1])))
            else:
                if inspect.isfunction(arg):
                    name = arg.__module__ + "." + arg.__name__
                    signatures.append(name + sep + str(inspect.signature(arg)))
        except ValueError:
            pass

    if _print:
        for sig in signatures:
            print(sig)

        return None

    return signatures
