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
"""Git wrapper"""

import functools
import itertools
import os
import re
import subprocess
from pathlib import Path
from typing import *


"""
Synopsis of `git status values <https://git-scm.com/docs/git-status>`_::

    X          Y     Meaning
    -------------------------------------------------
             [AMD]   not updated
    M        [ MD]   updated in index
    A        [ MD]   added to index
    D                deleted from index
    R        [ MD]   renamed in index
    C        [ MD]   copied in index
    [MARC]           index and work tree matches
    [ MARC]     M    work tree changed since index
    [ MARC]     D    deleted in work tree
    [ D]        R    renamed in work tree
    [ D]        C    copied in work tree
    -------------------------------------------------
    D           D    unmerged, both deleted
    A           U    unmerged, added by us
    U           D    unmerged, deleted by them
    U           A    unmerged, added by them
    D           U    unmerged, deleted by us
    A           A    unmerged, both added
    U           U    unmerged, both modified
    -------------------------------------------------
    ?           ?    untracked
    !           !    ignored
    -------------------------------------------------
"""


class GitRepository:
    """Accessor to certain information about a git repository

    Synopsis:
        Static methods:
            git_available        is_git_directory
        Properties:
            _permutations        staged              renamed
            _git                 unstaged            copied
            commit               tracked             unmerged
            digest               untracked           ignored
            status               added               modified
            dirty                deleted
        Methods:
            parse_status         _parse_status
    """

    def __init__(self, directory: Path = Path(".")) -> None:
        """
        Args:
            directory (Path, optional): a path to the working directory

        Raises:
            RuntimeError: if git binary is not available
            TypeError: if working directory is not valid
        """
        if not self.git_available():
            raise RuntimeError("git binary not available!")

        self._work_tree = directory
        self._git_dir = directory / ".git"

        if not self._work_tree.is_dir():
            raise TypeError(f"{self._work_tree!r} directory does not exist")

        if not self._git_dir.is_dir():
            raise TypeError(f".git directory missing in {self._work_tree!r}")

        if subprocess.run([*self._git, "status", "-s"], capture_output=True).returncode != 0:
            raise TypeError(f"not a git repository: {self._git_dir}")

        self._tracked_statuses = "MADRCU "
        self._untracked_statuses = "?!"

        # frozensets are used to allow cacheing of values
        self._valid_permutations = frozenset(
            {
                *("".join(x) for x in itertools.product(self._tracked_statuses, repeat=2)),
                "??",
                "!!",
            }
        )
        self._staged_permutations = frozenset(
            {_ for _ in self._valid_permutations if _[0] in "MADRCU"}
        )
        self._unstaged_permutations = frozenset(
            {_ for _ in self._valid_permutations if _[1] in "MADRCU"}
        )
        self._untracked_permutations = frozenset({"??", "!!"})
        self._tracked_permutations = self._valid_permutations.difference(
            self._untracked_permutations
        )

    @property
    def _permutations(self):
        return {
            "staged": self._staged_permutations,
            "unstaged": self._unstaged_permutations,
            "tracked": self._tracked_permutations,
            "untracked": self._untracked_permutations,
            "valid": self._valid_permutations,
        }

    @property
    def _git(self) -> List[str]:
        """Get a list of git commands to work in a specific directory"""
        return ["git", "--git-dir", str(self._git_dir), "--work-tree", str(self._work_tree)]

    @staticmethod
    def git_available() -> bool:
        """Is git available?"""
        return subprocess.run(["which", "git"]).returncode == 0

    @staticmethod
    def is_git_directory(directory: Path) -> bool:
        """Is a git directory?"""
        try:
            GitRepository(directory)
            return True
        except (RuntimeError, TypeError):
            return False

    @property
    def commit(self) -> str:
        """Get current git commit"""
        return subprocess.run(
            [*self._git, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()

    @property
    def digest(self) -> str:
        """Get current git commit's short digest"""
        return subprocess.run(
            [*self._git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()

    @property
    def status(self) -> List[Dict[str, str]]:
        """Get formatted output of `git status -s`"""
        r = (
            subprocess.run([*self._git, "status", "--short"], capture_output=True, text=True)
            .stdout.strip()
            .split(os.linesep)
        )
        return [{"status": rr[:2], "file": rr[3:]} for rr in r]

    def parse_status(
        self, statuses: Union[Set, List] = set(), file: AnyStr = r".+"
    ) -> List[Dict[str, str]]:
        """Wrapper for :meth:`_parse_status`"""
        statuses = frozenset(statuses) if statuses else self._valid_permutations
        return self._parse_status(statuses, file)

    @functools.lru_cache(maxsize=64)
    def _parse_status(
        self, statuses: frozenset = frozenset(), file: AnyStr = r".+"
    ) -> List[Dict[str, str]]:
        """Parse git status with statuses and file matching"""
        if not statuses.issubset(self._valid_permutations):
            raise ValueError(f"invalid status characters: {statuses!r}")
        regex = re.compile(file)

        return [
            entry
            for entry in self.status
            if entry["status"] in statuses and regex.match(entry["file"])
        ]

    @property
    def staged(self):
        return self.parse_status(statuses=self._staged_permutations)

    @property
    def unstaged(self):
        return self.parse_status(statuses=self._unstaged_permutations)

    @property
    def tracked(self):
        return self.parse_status(statuses=self._tracked_permutations)

    @property
    def untracked(self):
        return self.parse_status(statuses=self._untracked_permutations)

    @property
    def added(self):
        statuses = {_ for _ in self._valid_permutations if "A" in _}
        return self.parse_status(statuses=statuses)

    @property
    def deleted(self):
        statuses = {_ for _ in self._valid_permutations if "D" in _}
        return self.parse_status(statuses=statuses)

    @property
    def renamed(self):
        statuses = {_ for _ in self._valid_permutations if "R" in _}
        return self.parse_status(statuses=statuses)

    @property
    def copied(self):
        statuses = {_ for _ in self._valid_permutations if "C" in _}
        return self.parse_status(statuses=statuses)

    @property
    def unmerged(self):
        statuses = {_ for _ in self._valid_permutations if "U" in _}
        return self.parse_status(statuses=statuses)

    @property
    def ignored(self):
        return self.parse_status(statuses={"!!"})

    @property
    def modified(self):
        statuses = {_ for _ in self._valid_permutations if "M" in _}
        return self.parse_status(statuses=statuses)

    @property
    def dirty(self):
        """Does the git directory contain modified or untracked python files?"""
        regex = re.compile(r".+\.py$")
        files = [x for x in [*self.modified, *self.untracked] if regex.match(x["file"])]

        return True if files else False
