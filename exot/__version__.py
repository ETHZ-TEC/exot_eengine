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
