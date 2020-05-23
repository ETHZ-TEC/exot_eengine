"""Concrete ssh backends"""

import os
import typing as t
from pathlib import Path

import fabric

from exot.exceptions import *
from exot.util.file import check_access
from exot.util.logging import get_root_logger
from exot.util.misc import list2cmdline, validate_helper

from ._backend import Backend

__all__ = ("SSHBackend",)


class SSHBackend(Backend):
    @property
    def required_config_keys(self) -> t.List[str]:
        return ["ip", "user", "port", "key"]

    def validate(self) -> t.NoReturn:
        """Implements `validate` from the Configurable base class"""
        validate_helper(self.config, "ip", str)
        validate_helper(self.config, "user", str)
        validate_helper(self.config, "port", int)
        validate_helper(self.config, "key", str, Path)
        validate_helper(self.config, "gateway", type(None), str)

    def can_connect(self) -> bool:
        if not self.configured:
            get_root_logger().critical("Driver not configured!")
            return False

        try:
            self.keyfile = self.config.key
        except MisconfiguredError:
            get_root_logger().critical("Could not find key file " + str(self.config.key))
            return False

        return True

    @property
    def keyfile(self):
        return getattr(self, "_keyfile", None)

    @keyfile.setter
    def keyfile(self, value):
        _ = Path(os.path.expandvars(value)).expanduser()
        if not _.is_file():
            raise MisconfiguredError("key does not exist", _)
        if not check_access(_):
            raise MisconfiguredError("key not accessible", _)

        self._keyfile = _

    def __getstate__(self):
        # Custom getstate prevents the concrete keyfile being serialised. If key keyfile
        # location resolves to an absolute path, do not serialise it, but rather restore
        # one from the config.
        if self.keyfile and self.keyfile.is_absolute():
            self._keyfile = None
        if hasattr(self, "_fabric_config"):
            self._fabric_config = None

    def __setstate__(self, value):
        if not getattr(self, "_keyfile", None):
            self._keyfile = self.config.key

    @property
    def gateway(self) -> t.Optional[fabric.Connection]:
        if self.connection:
            return self.connection.gateway
        else:
            return None

    @property
    def connected(self) -> bool:
        return self.connection.is_connected if self.connection else False

    def connect(self) -> None:
        if not self.can_connect():
            if not self.config:
                raise CouldNotConnectError("config missing!")
            raise CouldNotConnectError("config invalid!")

        current = dict(
            host=self.config.ip,
            user=self.config.user,
            port=self.config.port,
            connect_kwargs=dict(key_filename=str(self.keyfile)),
            connect_timeout=2,
            gateway=fabric.Connection(os.path.expandvars(self.config.gateway))
            if "gateway" in self.config
            else None,
        )

        if getattr(self, "_fabric_config", {}) != current:
            self._fabric_config = current
            self._connection = fabric.Connection(**self._fabric_config)
            self.connection.open()
        else:
            if not self.connection.is_connected:
                self.connection.open()

        assert self.connection, "backend must have a working underlying connection"
        self._connected = self.connection.is_connected

        if not self.connected:
            raise CouldNotConnectError(current)

    def disconnect(self) -> None:
        assert self.connected, "needs to be connected before disconnecting"
        self.connection.close()
        self._connected = False

    @property
    def connection(self) -> t.Optional[fabric.Connection]:
        return getattr(self, "_connection", None)

    def _wrapper(
        self, f: t.Callable, cmd: Backend.CommandT, **kwargs: t.Any
    ) -> Backend.ReturnT:
        if not self.connection:
            raise NotConnectedError("not connected", self)
        if f not in [self.connection.run, self.connection.sudo]:
            raise TypeError(f"only 'run' and 'sudo' are accepted, got {f}")

        if not isinstance(cmd, (str, t.List)):
            raise TypeError(f"'cmd' ({type(cmd)}) not str or list")
        if isinstance(cmd, t.List):
            cmd = list2cmdline(cmd)

        # hide: do not output to console
        if "hide" not in kwargs:
            kwargs["hide"] = True
        # warn: do not throw when return code != 0
        if "warn" not in kwargs:
            kwargs["warn"] = True

        if "history" in kwargs:
            history = kwargs.pop("history")
            if not isinstance(history, bool):
                _ = f"keyword argument 'history' should be bool, got: {type(history)}"
                raise TypeError(_)
        else:
            history = True

        _ = f(cmd, **kwargs)
        _.stdout = _.stdout.rstrip("\r\n")  # strip trailing newlines
        _.stderr = _.stderr.rstrip("\r\n")  # strip trailing newlines

        if history:
            self._history.append(_)

        return _

    def run(self, cmd: Backend.CommandT, **kwargs: t.Any) -> Backend.ReturnT:
        return self._wrapper(f=self.connection.run, cmd=cmd, **kwargs)

    def sudo(self, cmd: Backend.CommandT, **kwargs: t.Any) -> Backend.ReturnT:
        return self._wrapper(f=self.connection.sudo, cmd=cmd, **kwargs)
