"""User prompts"""

import functools
import signal
import typing as t
from pathlib import Path
from re import match
from types import SimpleNamespace

from .file import check_access

__all__ = (
    "prompt",
    "prompt_path",
    "prompt_path_read",
    "prompt_path_write",
    "prompt_int",
    "prompt_float",
)


class ResponseHandlers(SimpleNamespace):
    def _str(response: str) -> str:
        return response

    def _bool(response: str) -> bool:
        return True if match(r"y(es)?", response.lower()) else False

    def _int(response: str) -> int:
        as_int: int

        try:
            as_int = int(response)
        except ValueError as Error:
            if len(Error.args) > 0:
                Error.args = (f"ResponseHandlers: {Error.args[0]}",)
            raise

        return as_int

    def _float(response: str) -> int:
        as_float: float

        try:
            as_float = float(response)
        except ValueError as Error:
            if len(Error.args) > 0:
                Error.args = (f"ResponseHandlers: {Error.args[0]}",)
            raise

        return as_float

    def _path(response: str, modes=["w"]) -> Path:
        as_path = Path(response)
        valid = True

        if isinstance(modes, str):
            valid = check_access(as_path, modes)
        elif isinstance(modes, t.List):
            for mode in modes:
                valid &= check_access(as_path, mode)

        if not valid:
            raise ValueError(f"path {as_path} is not accessible with provided modes {modes!r}")
        else:
            return as_path


def prompt(
    msg: str,
    expect: t.Optional[t.Type] = bool,
    _timeout: t.Optional[int] = None,
    _prompt: str = "> ",
    _ask: str = "? ",
    _handler: t.Optional[t.Callable[[str], t.Any]] = None,
    _fail: bool = False,
    **kwargs,
):
    """prompt user for input, with timeouts and type handling
    """
    if expect and not isinstance(expect, type):
        raise TypeError(f"'expect' should be a type, got: {type(expect)}")

    if not _handler:
        if expect is bool:
            _handler = ResponseHandlers._bool
            _ask = " (y/[n])? " if _ask != "? " else _ask
        elif expect is Path:
            _handler = ResponseHandlers._path
        elif expect is int:
            _handler = ResponseHandlers._int
        elif expect is float:
            _handler = ResponseHandlers._float
        else:
            _handler = ResponseHandlers._str
    else:
        if not isinstance(_handler, t.Callable):
            raise TypeError("'_handler' should be callable")

    class AlarmException(Exception):
        pass

    def signal_handler(number: int, frame):
        assert number == signal.SIGALRM.value, f"unexpected signal: {number}"
        raise AlarmException("timed out!")

    display = f"{_prompt} {msg}{_ask}"

    if _timeout:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(_timeout)

    try:
        response = input(display)
    except EOFError:
        if expect is bool:
            # If Enter is pressed on the prompt, consider it a
            response = "n"
        else:
            raise
    except AlarmException:
        if _fail:
            raise TimeoutError("waiting for user input timed out!")
        else:
            return None
    else:
        if _timeout:
            signal.signal(signal.SIGALRM, signal.SIG_IGN)
            signal.alarm(0)
        return _handler(response, **kwargs)


"""prompt with a path handler"""
prompt_path = functools.partial(
    prompt, expect=Path, _handler=ResponseHandlers._path, permissions=["r", "w"]
)

"""prompt with a path handler -> read access"""
prompt_path_read = functools.partial(
    prompt, expect=Path, _handler=ResponseHandlers._path, permissions="r"
)

"""prompt with a path handler -> write access"""
prompt_path_write = functools.partial(
    prompt, expect=Path, _handler=ResponseHandlers._path, permissions="w"
)

"""prompt for integer values"""
prompt_int = functools.partial(prompt, expect=int, _handler=ResponseHandlers._int)

"""prompt for float values"""
prompt_float = functools.partial(prompt, expect=float, _handler=ResponseHandlers._float)
