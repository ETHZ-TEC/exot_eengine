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
