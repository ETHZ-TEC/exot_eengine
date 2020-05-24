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
"""Covert channel analysis mixin classes."""

import abc

__all__ = ("CapacityContinuous", "PerformanceSweep")


class CapacityContinuous(metaclass=abc.ABCMeta):
    """Handler to perform the capacity calculation for a continuous (covert) channel."""

    def _execute_handler(self, *args, **kwargs):
        """Calcualate the capacity for a specified environment."""
        for keyword in ["phase", "env", "rep", "window_size", "matcher"]:
            if keyword not in kwargs:
                raise ValueError("kwargs keyword %s missing!" % (keyword,))
        self.experiment.generate_spectra(
            phases=[kwargs["phase"]],
            envs=[kwargs["env"]],
            reps=[kwargs["rep"]],
            window_size=kwargs["window_size"],
            io={"matcher": kwargs["matcher"]},
        )
        Sqq = self.experiment.spectrum_as_matrix("Sqq", phase, env, 0)
        Shh = self.experiment.spectrum_as_matrix("Shh", phase, env, 0)
        Sxx = self.experiment.spectrum_as_matrix("Sxx", phase, env, 0)
        Syy = self.experiment.spectrum_as_matrix("Syy", phase, env, 0)
        p0 = wrangle.filter_data(
            self.experiment.p0, phase=phase, environment=env, repetition=0
        )["p0"].values[0]
        return dict(
            C_wf=capacity.classic_waterfilling(p0, Sqq, Shh),
            C_cwf=capacity.constrained_waterfilling(p0, Sqq, Shh),
        )


class PerformanceSweep(metaclass=abc.ABCMeta):
    # At the moment, this functionality is in the performance.py experiment class in the calculate_performance_metrics method.
    pass
