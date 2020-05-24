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
"""Misc helpers"""
import numpy as np
import pandas as pd

from exot.util.attributedict import LabelMapping


def all_labels_in_dataset(experiment, sort_str=False, **kwargs):
    for key in ["phase", "labelcolumn", "io"]:
        if key not in kwargs.keys():
            raise ValueError(f"key {key} not specified in kwargs!")
    ingest_args = kwargs.copy()
    labels = np.array([])
    # for rep in range(1):
    for rep in range(experiment.config.EXPERIMENT.PHASES[ingest_args["phase"]].repetitions):
        ingest_args["io"]["rep"] = rep
        # for cur_run in [experiment.phases[ingest_args['phase']]['antutu2']]:
        for cur_run in experiment.phases[ingest_args["phase"]].values():
            try:
                cur_run.ingest(**ingest_args)
                labels = np.concatenate(
                    (labels, cur_run.i_rawstream[ingest_args["labelcolumn"]].unique())
                )
            except:
                print("Could not ingest run", cur_run)

    labels = pd.DataFrame(labels)[0].unique().flatten()
    if sort_str:
        return _sort_str_labels(labels)
    else:
        return labels


def sort_str_labels(labels):
    convert = lambda x: str(x)
    return np.array(list(map(convert, labels)))


def generate_unique_labels_mapping(labels_keys, labels_str):
    if labels_str is None:
        for key in labels_keys:
            if key not in labels_str.keys():
                labels_str[key] = key
        labels_map = dict(
            [
                (y, {"int": x, "str": str(labels_keys[x])})
                for x, y in enumerate(set(labels_keys))
            ]
        )
    else:
        labels_map = dict(
            [(y, {"int": x, "str": labels_str[x]}) for x, y in enumerate(set(labels_keys))]
        )
    return LabelMapping(labels_map)
