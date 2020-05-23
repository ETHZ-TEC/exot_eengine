"""Mixins for the different layers"""
import copy
import typing as t

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.signal

from exot.exceptions import *
from exot.util.misc import get_valid_access_paths, getitem, is_scalar_numeric, get_cores_and_schedules

class RDPmixins():
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

