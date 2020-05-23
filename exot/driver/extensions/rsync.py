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
import typing as t
from pathlib import Path
from shlex import quote

from exot.util.file import _path_type_check, check_access

from .._driver import Driver
from .._mixins import FileMethodsMixin, TransferMethodsMixin


class TransferMethods(TransferMethodsMixin, FileMethodsMixin):
    def fetch(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_FETCH_EXCLUDE_LIST,
    ) -> None:
        path_to = _path_type_check(path_to)
        path_from = str(path_from)
        assert check_access(path_to, "w"), "path_to must be writable"
        assert self.exists(path_from), "remote path_from must exist"

        if self.is_dir(path_from):
            path_to = str(path_to) + "/"
            path_from += "/"

        if not isinstance(exclude, t.List):
            raise TypeError("'exclude' must be a list")

        for item in exclude:
            if not isinstance(item, str):
                raise TypeError("'exclude' items must be of type 'str'")

        rsync_args = {
            "key": self.backend.keyfile,
            "port": self.backend.config["port"],
            "user": self.backend.config["user"],
            "host": self.backend.config["ip"],
            "from": path_from,
            "to": path_to,
            "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude]) if exclude else "",
        }

        if "gateway" in self.backend.config:
            rsync_args.update(gateway=self.backend.config["gateway"])
            command = (
                # "rsync -a -e \"ssh -o 'ProxyCommand nohup ssh -q -A {gateway} nc -q0 %h %p' "
                "rsync -a -e \"ssh -o 'ProxyCommand ssh -q -A {gateway} -W %h:%p' "
                '-i {key} -p {port}" {exclude} {user}@{host}:{from} {to}'
            )
        else:
            command = 'rsync -a -e "ssh -i {key} -p {port}" {exclude} {user}@{host}:{from} {to}'

        command = command.format(**rsync_args)
        if hasattr(self.backend.connection, "local"):
            return self.backend.connection.local(command, hide=True, warn=True)
        else:
            return self.backend.connection(command, hide=True, warn=True)

    def send(
        self,
        path_from: t.Union[Path, str],
        path_to: t.Union[Path, str],
        exclude: t.List[str] = Driver.DEFAULT_SEND_EXCLUDE_LIST,
    ) -> None:
        path_to = str(path_to)
        path_from = _path_type_check(path_from)
        assert path_from.exists(), "path_from must exist"

        # Copying directories with different combinations of trailing slashes may yield
        # unexpected results. If the `path_from` is a directory, append trailing slashes
        # to both `path_from` and `path_to`.
        if path_from.is_dir():
            path_from = str(path_from) + "/"
            path_to += "/"

        if not self.access(path_to, "w"):
            self.mkdir(str(Path(path_to).parent), parents=True)

        if not isinstance(exclude, t.List):
            raise TypeError("'exclude' must be a list")

        for item in exclude:
            if not isinstance(item, str):
                raise TypeError("'exclude' items must be of type 'str'")

        rsync_args = {
            "key": self.backend.keyfile,
            "port": self.backend.config["port"],
            "user": self.backend.config["user"],
            "host": self.backend.config["ip"],
            "from": path_from,
            "to": path_to,
            "exclude": " ".join([f"--exclude {quote(_)}" for _ in exclude]) if exclude else "",
        }

        if "gateway" in self.backend.config:
            rsync_args.update(gateway=self.backend.config["gateway"])
            command = (
                # "rsync -a -e \"ssh -o 'ProxyCommand nohup ssh -q -A {gateway} nc -q0 %h %p' "
                "rsync -a -e \"ssh -o 'ProxyCommand ssh -q -A {gateway} -W %h:%p' "
                '-i {key} -p {port}" {exclude} {from} {user}@{host}:{to}'
            )
        else:
            command = 'rsync -a -e "ssh -i {key} -p {port}" {exclude} {from} {user}@{host}:{to}'

        command = command.format(**rsync_args)
        if hasattr(self.backend.connection, "local"):
            return self.backend.connection.local(command, hide=True, warn=True)
        else:
            return self.backend.connection(command, hide=True, warn=True)
