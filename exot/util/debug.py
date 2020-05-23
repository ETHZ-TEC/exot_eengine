# Copyright (c) 2015-2020, Swiss Federal Institute of Technology (ETH Zurich)
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
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
