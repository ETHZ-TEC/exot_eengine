from __future__ import annotations

import typing as t

__all__ = ("Process",)


class Process:
    def __init__(
        self,
        driver: object,
        invocation: object,
        identity: t.Union[int, str],
        slaves: t.List[Process] = [],
        duration: t.Optional[float] = None,
    ):
        """Creates a Process instance

        Args:
            driver (object): The driver
            invocation (object): The command invocation (invoke_result)
            identity (t.Union[int, str]): The identity, for example the PID or Component name
            slaves (t.List[Process], optional): The slave processes
        """
        self.driver = driver
        self.invocation = invocation
        self.identity = identity
        self.slaves = slaves
        self.duration = duration

        self.update()

    def __repr__(self) -> str:
        return "<Process {} of {!r} at {}>".format(self.identity, self.driver, hex(id(self)))

    def __hash__(self) -> int:
        return hash(self.identity)

    @property
    def exited(self) -> t.Optional[int]:
        """Gets the exit code of the process

        Returns:
            t.Optional[int]: The exit code or None if failed or running
        """
        self.update()
        return self.invocation.exited

    @property
    def stderr(self) -> t.Optional[str]:
        """Gets the invocation's standard error

        Returns:
            t.Optional[str]: The standard error
        """
        self.update()
        return self.invocation.stderr

    @property
    def stdout(self) -> t.Optional[str]:
        """Gets the invocation's standard output

        Returns:
            t.Optional[str]: The standard output
        """
        self.update()
        return self.invocation.stdout

    @property
    def children(self) -> t.List[t.Union[int, str]]:
        """Gets the identities of child properties

        Returns:
            t.List[t.Union[int, str]]: The children identities
        """
        if hasattr(self.invocation, "children"):
            return self.invocation.children or []
        else:
            return []

    @property
    def slaves(self) -> t.List[Process]:
        """Gets the slave processes

        Returns:
            t.List[Process]: The slave processes
        """
        return self._slaves

    @slaves.setter
    def slaves(self, value: t.List[Process]) -> None:
        """Sets the slave processes

        Args:
            value (t.List[Process]): The list of slave processes

        Raises:
            TypeError: Wrong type(s) supplied
        """
        if not isinstance(value, t.List):
            raise TypeError()
        if any(not isinstance(_, Process) for _ in value):
            raise TypeError()

        self._slaves = value

    def update(self):
        """Updates the process status
        """
        if hasattr(self.invocation, "update"):
            self.invocation.update()
