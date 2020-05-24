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
#!/usr/bin/env bash
u:info() { (echo >&2 -e "[\e[1m\e[34minfo\e[0m]\t" "$@"); }
u:warn() { (echo >&2 -e "[\e[1m\e[33mwarn\e[0m]\t" "$@"); }
u:erro() {
    (echo >&2 -e "[\e[1m\e[31merror\e[0m]\t" "$@")
    exit 1
}

PORT=${1:-8889}
PYENV=${2:-toolkit}

# Only execute this script from within the project directory tree
if test ! -e "$PWD/exot"; then
    u:erro "This script is meant to be run with the project directory as working directory"
fi

pyenv activate "$PYENV" || u:erro "pyenv activation of '$PYENV' failed"
poetry install || u:erro "poetry environment installation failed; you can try deleting the cache with 'rm -rf $HOME/.cache/pypoetry'"
ipython kernel install --user --name="$PYENV" || u:erro "iPython kernel installation failed"
jupyter notebook --no-browser --port="$PORT" || u:erro "starting jupyter notebook server failed"
