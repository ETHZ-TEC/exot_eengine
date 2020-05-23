"""Filesystem utilities"""

import os
import shutil
import typing as t
from datetime import datetime
from pathlib import Path

import invoke

from exot.exceptions import WrongRootPathError
from exot.util.misc import random_string

__all__ = (
    "check_access",
    "validate_root_path",
    "move",
    "copy",
    "delete",
    "add_random",
    "add_timestamp",
)


def _path_type_check(path: t.Union[Path, str], desc: str = "path") -> Path:
    """Normalise a path to a Path type"""
    assert isinstance(desc, str)
    if not isinstance(path, (str, Path)):
        raise TypeError(f"wrong type supplied for '{desc}'", path)
    if not isinstance(path, Path):
        path = Path(path)

    return path


def check_access(path: t.Union[Path, str], mode: str = "r") -> bool:
    """Check access permissions on a path"""
    path = _path_type_check(path)

    if mode == "w":
        if not path.exists():
            # If file/directory doesn't exist, check if parent can be written
            return os.access(path.parent, os.W_OK)
        return os.access(path, os.W_OK)
    elif mode == "r":
        return os.access(path, os.R_OK)
    elif mode == "x":
        return os.access(path, os.X_OK)
    else:
        raise ValueError(f"mode can be only one of {['w', 'r', 'x']!r}")


def validate_root_path(path: t.Union[Path, str], throw: bool = True) -> bool:
    """Check if a root path is a valid exot root path"""
    path = _path_type_check(path)

    valid = True
    valid &= (path / "pyproject.toml").exists()
    valid &= (path / "exot").is_dir()
    valid &= (path / "environments").is_dir()

    if not valid and throw:
        raise WrongRootPathError(path)

    return valid


def add_random(path: t.Union[Path, str], length: int = 5) -> Path:
    """Add a random suffix to a path"""
    path = _path_type_check(path)

    return path.parent / f"{path.stem}_{random_string(length)}{path.suffix}"


def add_timestamp(path: t.Union[Path, str], time: t.Optional[str] = None) -> Path:
    """Add a timestamp to a path"""
    path = _path_type_check(path)

    valid_times = ["accessed", "modified", "created"]
    if time and time not in valid_times:
        raise ValueError(f"'time' should be one of {valid_times!r}")

        _ = path.stat()
        select = dict(zip(valid_times, [_.st_atime, _.st_mtime, _.st_ctime]))
        timestamp = datetime.fromtimestamp(select[time]).isoformat("_", timespec="seconds")
    else:
        timestamp = datetime.now().isoformat("_", timespec="seconds")
        timestamp = timestamp.replace(":", "-")

    return path.parent / f"{path.stem}_{timestamp}{path.suffix}"


def move(path: t.Union[Path, str], to: t.Union[Path, str]) -> Path:
    """Move a file or a directory"""
    path = _path_type_check(path)
    to = _path_type_check(to)

    if not path.exists():
        raise FileNotFoundError(path)

    if to.exists():
        raise FileExistsError(to)

    return shutil.move(path, to)


def copy(path: t.Union[Path, str], to: t.Union[Path, str], replace=False) -> Path:
    """Copy a file or a directory"""
    path = _path_type_check(path)
    if not isinstance(to, (str, Path)):
        raise TypeError("wrong type supplied for 'to'", to)
    if not isinstance(to, Path):
        to = Path(to)

    if not path.exists():
        raise FileNotFoundError(path)

    if to.exists() and not replace:
        raise FileExistsError(to)

    if path.is_file():
        return shutil.copy2(path, to)
    else:
        return shutil.copytree(path, to)


def delete(path: t.Union[Path, str]) -> None:
    """Delete a file or a directory"""
    path = _path_type_check(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        os.remove(path)
    else:
        shutil.rmtree(path)


def backup(path: t.Union[Path, str], config: t.Mapping) -> invoke.runners.Result:
    _path = _path_type_check(path)
    if not _path.exists():
        raise FileNotFoundError(path)
    if not _path.is_file():
        raise ValueError("util.file.backup can only backup files, not directories")

    if [k for k in ["user", "host", "key", "port", "path"] if k not in config]:
        raise ValueError("util.file.backup: misconfigured backup settings")

    config["from"] = _path
    config["key"] = Path(os.path.expandvars(config["key"])).expanduser()

    command = "scp -i {key} -P {port} {from} {user}@{host}:{path}".format(**config)
    return invoke.run(command, hide=True)
