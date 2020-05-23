"""Context manager for timeouts"""

import contextlib
import signal

__all__ = ("Timeout",)


class DataproTimeoutError(Exception):
    pass


class Timeout(contextlib.ContextDecorator):

    """A timeout context decorator/manager using POSIX signal's

    Caveats:
        Can only run in the main thread due to using POSIX signals!

    Attributes:
        duration (int): the timeout duration in seconds
        throwing (bool): does it throw when timed out?
        timed_out (bool): has a time out occured?

    Examples:
        >>> from time import sleep

        >>> with Timeout(1):
        >>>    sleep(1.2)
        >>>    print("This message is never printed!")

        >>> with Timeout(1) as tm:
        >>>     sleep(1.2)
        >>> tm.timed_out
        True

        >>> @Timeout(1)
        >>> def function_with_timeout(seconds: float):
        >>>     sleep(seconds)
                return 0
        >>> function_with_timeout(0.5)
        0
        >>> function_with_timeout(1.5)
        None
    """

    def __init__(self, duration: int, throwing: bool = False):
        self.duration = int(duration)  # type: int
        self.throwing = throwing  # type: bool
        self.timed_out = False  # type: bool

    def __repr__(self):
        return "<Timeout at {}, {}, {}>".format(
            hex(id(self)),
            "throwing" if self.throwing else "non-throwing",
            "timed out" if self.timed_out else "not timed out",
        )

    def __handler__(self, number: int, frame):
        assert number == signal.SIGALRM.value, f"unexpected signal: {number}"
        raise DataproTimeoutError(f"Timed out after {self.duration} seconds")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.__handler__)
        signal.alarm(self.duration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(0)

        if exc_type and exc_type is DataproTimeoutError:
            self.timed_out = True
        elif exc_type and exc_type is not DataproTimeoutError:
            raise exc_type(exc_val, exc_tb)

        if self.timed_out and self.throwing:
            raise exc_type(exc_val, exc_tb)
        else:
            return self.timed_out
