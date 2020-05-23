"""Context manager for timeouts

Attributes:
    Timeout (ContextDecorator): The default timeout class (ThreadTimeout)
"""

import signal
from contextlib import ContextDecorator
from threading import Condition, Thread
from types import FrameType, TracebackType
from typing import Any, NoReturn, Union

__all__ = ("SignalTimeout", "ThreadTimeout", "Timeout")


class TimeoutException(Exception):
    pass


class SignalTimeout(ContextDecorator):

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
        self.duration: int = int(duration)
        self.throwing: bool = throwing
        self.timed_out: bool = False

    def __repr__(self) -> str:
        return "<Timeout at {}, duration {}, {}, {}>".format(
            hex(id(self)),
            self.duration,
            "throwing" if self.throwing else "non-throwing",
            "timed out" if self.timed_out else "not timed out",
        )

    def __handler__(self, number: int, frame: FrameType):
        assert number == signal.SIGALRM.value, f"unexpected signal: {number}"
        raise TimeoutException(f"Timed out after {self.duration} seconds")

    def __enter__(self) -> object:
        signal.signal(signal.SIGALRM, self.__handler__)
        signal.alarm(self.duration)
        return self

    def __exit__(
        self, exc_type: type, exc_val: Any, exc_tb: TracebackType
    ) -> Union[NoReturn, bool]:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(0)

        if exc_type and exc_type is TimeoutException:
            self.timed_out = True
        elif exc_type and exc_type is not TimeoutException:
            raise exc_type(exc_val).with_traceback(exc_tb)

        if self.timed_out and self.throwing:
            raise exc_type(exc_val).with_traceback(exc_tb)
        else:
            return self.timed_out


class ThreadTimeout(ContextDecorator):

    """Timeout context manager and decorator implemented with threads and condition variables

    Attributes:
        duration (bool): The timeout duration in seconds
        throwing (bool):  Should throw on timeout?
        timed_out (bool): Is timed out?
        _cv_r (bool): The outcome on condition variable wait
        _cv (Condition): The condition variable
        _thread (Thread): The timeout/waiting thread
    """

    def __init__(self, duration: float, throwing: bool = False):
        """Initialises the timeout context decorator

        Args:
            duration (float): The timeout duration in seconds
            throwing (bool, optional): Should throw in timeout?
        """
        self.duration: bool = float(duration)
        self.throwing: bool = throwing
        self.timed_out: bool = False
        self._cv_r: bool = False
        self._cv: Condition = Condition()
        self._thread: Thread = Thread(target=self.__handler__)

    def __repr__(self) -> str:
        return "<ThreadTimeout at {}, duration {}, {}, {}>".format(
            hex(id(self)),
            self.duration,
            "throwing" if self.throwing else "non-throwing",
            "timed out" if self.timed_out else "not timed out",
        )

    def __handler__(self) -> None:
        with self._cv:
            self._cv_r = self._cv.wait(timeout=self.duration)

    def __enter__(self) -> object:
        self._thread.start()
        return self

    def __exit__(self, exc_type: type, exc_val: Any, exc_tb: TracebackType):
        with self._cv:
            self._cv.notify()
        self._thread.join()

        if not self._cv_r:
            self.timed_out = True

        elif exc_type and exc_type is not TimeoutException:
            raise exc_type(exc_val).with_traceback(exc_tb)

        if self.timed_out and self.throwing:
            raise TimeoutException(repr(self)).with_traceback(exc_tb)

        return self.timed_out


Timeout = ThreadTimeout
