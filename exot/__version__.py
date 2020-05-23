from pathlib import Path

try:
    import toml
except ImportError as e:
    from sys import stderr

    print(
        "Failed to import a module defined in pyproject.toml\n"
        "Make sure to execute inside a proper venv, e.g. using `poetry run`.",
        file=stderr,
    )
    raise e

# Get version from pyproject.toml
_in_here = Path("pyproject.toml")
_in_parent = Path("../pyproject.toml")

# Check if the pyproject.toml file exists
if _in_here.exists() or _in_parent.exists():
    try:
        __pyproject__ = toml.load(_in_here) if _in_here.exists() else toml.load(_in_parent)
    except (OSError, PermissionError):
        __pyproject__ = None

# Set __version__
try:
    __version__ = __pyproject__["tool"]["poetry"]["version"]
except (KeyError, TypeError):
    __version__ = "unknown"
finally:
    del _in_here, _in_parent
    del toml, Path
