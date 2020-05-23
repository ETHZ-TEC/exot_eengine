"""Concrete mixins for extended functionality"""

import abc
import typing as t
from shlex import quote

from exot.util.misc import list2cmdline, mro_hasattr, random_string, timestamp

from ._backend import CommandT, ReturnT
from ._driver import Driver


class NohupPersistenceMixin(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        """Initialises the mixin and checks if it's used in the right context

        Args:
            *args: Forwarded positional arguments
            **kwargs: Forwarded keyword arguments

        Raises:
            TypeError: When mixin is supplied to a class that is not a subclass of the driver
        """
        if Driver not in type(self).mro():
            raise TypeError("The NohupPersistenceMixin can only be used on Driver subclasses.")

        super().__init__(*args, **kwargs)

    def persistent(
        self, cmd: CommandT, chain: t.Optional[CommandT] = None, **kwargs: t.Any
    ) -> ReturnT:
        """Run a persistent command that is detached from the calling process

        In addition to the base ReturnT, the output contains attributes:
        -   pid: the pid of the detached process
        -   pgid: the process group id of the detached process
        -   children: child processes spawned by the shell

        Args:
            cmd (CommandT): the command to run persistently
            chain (t.Optional[CommandT], optional):
                a command to run once the primary command finishes
            **kwargs (t.Any): optional keyword arguments: "history", "command_only"
        """
        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")
        if isinstance(cmd, t.List):
            cmd = list2cmdline(cmd)

        if chain and not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'chain' ({type(chain)}) not str or list")
        if chain and isinstance(cmd, t.List):
            chain = list2cmdline(chain)

        if "history" in kwargs:
            history = kwargs.pop("history")
            if not isinstance(history, bool):
                _ = f"keyword argument 'history' should be bool, got: {type(history)}"
                raise TypeError(_)
        else:
            history = True

        # nohup temporary files
        nohup_id = timestamp() + "_" + random_string(5)
        nohup = {
            "pre": "/usr/bin/env bash -c",
            "id": nohup_id,
            "out": ".nohup-" + nohup_id + "-out",
            "err": ".nohup-" + nohup_id + "-err",
            "ret": ".nohup-" + nohup_id + "-ret",
            "rch": ".nohup-" + nohup_id + "-rch",
        }

        nohup["files"] = [nohup["out"], nohup["err"], nohup["ret"], nohup["rch"]]

        # echo the exit code to a unique temporary file
        nohup["wrap"] = quote(
            cmd
            + "; echo $? > {}{}".format(
                nohup["ret"], f"; {chain}; echo $? > {nohup['rch']}" if chain else ""
            )
        )
        # complete nohup + disown command
        _cmd = "nohup {pre} {wrap} 1>{out} 2>{err} & disown && echo $!".format(**nohup)

        if kwargs.get("command_only"):
            return _cmd

        invocation = self.command(_cmd, history=False)

        def _read():
            sep = random_string(5)
            line = " <(echo {}) ".format(sep)
            command = "cat " + line.join(nohup["files"])
            result = self.command(command)
            values = [_.strip("\n") for _ in result.stdout.split(sep)]
            assert len(values) == 4, "expected exactly 4 values after splitting"
            # for stderr, ignore the first line with contains "nohup: ignoring input"
            values[0] = values[0] if values[2] else None
            values[1] = "\n".join(values[1].split("\n")[1:]) if values[2] else None
            # return codes
            values[2] = int(values[2]) if values[2] else None
            values[3] = int(values[3]) if values[3] else None

            return dict(zip(["out", "err", "ret", "rch"], values))

        def _cleanup():
            self.command(["rm", *nohup["files"]])

        nohup["pid"] = int(invocation.stdout.strip())
        _pgid = self.command(f"ps -o pgid= {nohup['pid']}")
        _cpids = self.command(f"pgrep -P {nohup['pid']}")
        nohup["pgid"] = int(_pgid.stdout.strip()) if _pgid else None
        nohup["children"] = [int(_) for _ in _cpids.stdout.split("\n")] if _cpids else None
        nohup["runner"] = self.command
        nohup["command"] = _cmd
        nohup["chain"] = chain
        nohup["chain_exited"] = None

        nohup["read"] = _read
        nohup["cleanup"] = _cleanup
        nohup["cleaned_up"] = False

        invocation.pid = nohup["pid"]
        invocation.pgid = nohup["pgid"]
        invocation.children = nohup["children"]
        invocation.command = cmd
        invocation.persistent = nohup

        def update():
            assert hasattr(invocation, "persistent")
            values = invocation.persistent["read"]()

            if not invocation.persistent["cleaned_up"]:
                invocation.stdout = values["out"]
                invocation.stderr = values["err"]
                invocation.exited = values["ret"]
                invocation.persistent["chain_exited"] = values["rch"]

            if invocation.exited == 0 and not invocation.persistent["chain"]:
                invocation.persistent["cleanup"]()
                invocation.persistent["cleaned_up"] = True
            elif (
                invocation.exited == 0
                and invocation.persistent["chain"]
                and invocation.persistent["chain_exited"] == 0
            ):
                invocation.persistent["cleanup"]()
                invocation.persistent["cleaned_up"] = True

        invocation.cleanup = _cleanup
        invocation.update = update
        invocation.update()

        if history and hasattr(self.backend, "_history"):
            self.backend._history.append(invocation)

        return invocation
