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
import abc

from exot.channel.mixins.mldataset import GenericDataSetHandler
from exot.util.logging import get_root_logger

__all__ = "ThermalAppDecectionDataSetHandler"


class ThermalAppDecectionDataSetHandler(GenericDataSetHandler):
    def dataset_parameters_to_dict(self) -> dict:
        param_dict = GenericDataSetHandler.dataset_parameters_to_dict(self)
        param_dict["unknown_label"] = self.unknown_label
        param_dict["num_labels"] = self.num_labels
        param_dict["batch_size_timesteps"] = self.batch_size_timesteps
        return param_dict

    @property
    def unknown_label(self):
        return getattr(self, "_unknown_label", False)

    @unknown_label.setter
    def unknown_label(self, value: bool) -> None:
        setattr(self, "_unknown_label", value)

    @property
    def num_feature_dims(self):
        return self.config.DATASET.num_feature_dims

    @num_feature_dims.setter
    def num_feature_dims(self, value):
        self.config.DATASET.num_feature_dims = value

    @property
    def max_timesteps(self):
        return self.config.DATASET.max_timesteps

    @max_timesteps.setter
    def max_timesteps(self, value):
        self.config.DATASET.max_timesteps = value

    @property
    def batch_size_timesteps(self):
        return self.batch_size * self.max_timesteps

    @property
    def num_labels(self):
        return getattr(self, "_num_labels", 0)

    @num_labels.setter
    def num_labels(self, value: int) -> None:
        self._num_labels = value
