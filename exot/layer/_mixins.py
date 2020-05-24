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
"""Mixins for the different layers"""
import copy
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import (
    get_cores_and_schedules,
    get_valid_access_paths,
    getitem,
    is_scalar_numeric,
)


class RDPmixins:
    @property
    def cores_and_schedules(self):
        """Get the cores and schedules set of 2-tuples"""
        return self._cores_and_schedules

    @cores_and_schedules.setter
    def cores_and_schedules(self, value):
        """Set the cores and schedules set of 2-tuples"""
        try:
            assert isinstance(value, set), "must be a set"
            all(
                isinstance(_, tuple) and len(_) == 2 for _ in value
            ), "must contain only 2-tuples"
            all(
                isinstance(_[0], int) and isinstance(_[1], str) for _ in value
            ), "must contain 2-tuples of (int, str)"
        except AssertionError as e:
            raise LayerMisconfigured("cores_and_schedules {}, got {}".format(*e.args, value))

        self._cores_and_schedules = value
