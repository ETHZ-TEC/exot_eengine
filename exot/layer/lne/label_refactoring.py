"""Label refactioring layer"""
import abc
import typing as t

import math
import numpy as np

from exot.exceptions import *
from exot.util.attributedict import AttributeDict
from exot.util.wrangle import *

from .._base import Layer

class LabelRefactoring(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @property
    def requires_runtime_config(self) -> (bool, bool):
        """Does the layer's (encode, decode) require runtime configuration?"""
        return (False, True)

    @property
    def required_config_keys(self):
        """The required config keys

        Implements the `required_config_keys` from Configurable base class
        """
        return ["options", "label_columns", "env"]

    def validate(self) -> t.NoReturn:
        """Implementation of Configurable's `validate`"""
        try:
            if "label_columns" in self.config:
                assert isinstance(self.config.label_columns, list), ("label_columns", list, type(self.config.label_columns))
                for elem in self.config.label_columns:
                    assert isinstance(elem, int), ("label_columns element", int, type(elem))
                    assert (elem >= 0), "label_columns element negative"
            if "options" in self.config:
                assert isinstance(self.config.options, (dict, AttributeDict)), ("Options", (dict, AttributeDict), type(self.config.options))
                # TODO check options structure
                #assert len(self.config.options) == 3, "Options needs to have at three elements (str, list, object)"
                #assert isinstance(self.config.options[0], str), ("Options[0]", str, type(self.config.options[0]))
                #assert isinstance(self.config.options[1], list),    ("Options[1]", list, type(self.config.options[1]))
                #if self.config.options[0] == "extend":
                #    assert isinstance(self.config.options[2], tuple), ("Options[2]", tuple, type(self.config.options[2]))
                #    assert len(self.config.options[2]) == 2, "Options[2] has to contain two values"
                #    assert isinstance(self.config.options[2][0], int), ("Options[2]", int, type(self.config.options[2][0]))
                #    assert isinstance(self.config.options[2][1], int), ("Options[2]", int, type(self.config.options[2][1]))
        except AssertionError as e:
            raise MisconfiguredError("timevalue: {} expected {}, got: {}".format(*e.args[0]))

    def _encode(self, upper):
        """
        This layer is not intended for encoding, therefore a simple pass-through is implemented at this stage.
        """
        return upper

    def _decode(self, lnestream: np.ndarray) -> np.ndarray:
        """Decode an input line-encoded stream. This means that label refactoring needs to be applied. \
        Following options are available:
            * ("extend", labels, (front, back)) ... Extends all labels in labels.
                                                    If a label starts at timestamp s and ends at timestamp e,
                                                    after refactoring it will start at s-front and end at e+back.

        Args:
            lnestream (np.ndarray): a n-d array

        Returns:
            np.ndarray: a n-d array with refactored labels
        """
        symstream = lnestream.copy()

        if "replace" in self.config.options:
            for label in self.config.options['replace']['affected_labels']:
                for label_col_idx in self.config.label_columns:
                    # Assume the traces starts and ends with the same app...
                    if symstream[0, label_col_idx] in self.config.options['replace']['affected_labels'] and symstream[-1, label_col_idx] in self.config.options['replace']['affected_labels']:
                        # Find first label not to replace and set zero/last label
                        other_idxes = np.where(symstream[:, label_col_idx] != label)[0]
                        symstream[0:other_idxes[0], label_col_idx] = symstream[other_idxes[0], label_col_idx]

                        symstream[other_idxes[-1]+1:, label_col_idx] = symstream[other_idxes[-1], label_col_idx]
                    elif symstream[0, label_col_idx] in self.config.options['replace']['affected_labels']:
                        symstream[0, label_col_idx] = symstream[-1, label_col_idx]
                    elif symstream[-1, label_col_idx] in self.config.options['replace']['affected_labels']:
                        symstream[-1, label_col_idx] = symstream[0, label_col_idx]
                    labels_to_replace = np.where(symstream[:,label_col_idx] == label)[0]
                    if len(labels_to_replace) == 0:
                        continue
                    if self.config.options['replace']['with'] == "before" :
                        def replace_function(idx):
                            symstream[idx, label_col_idx] = symstream[idx-1, label_col_idx]
                    elif self.config.options['replace']['with'] == "after":
                        def replace_function(idx):
                            symstream[idx, label_col_idx] = symstream[idx+1, label_col_idx]
                        labels_to_replace[::-1].sort()
                    vec_replace_function = np.vectorize(replace_function)
                    vec_replace_function(labels_to_replace)

        if "smoothen" in self.config.options:
            for label_col_idx in self.config.label_columns:
                label_changes = np.where(np.diff(symstream[:,label_col_idx], axis=0) != 0)[0]
                for idx in range(len(label_changes[:-1])):
                    if symstream[label_changes[idx+1],0] - symstream[label_changes[idx],0] < self.config.options['smoothen']['threshold_s']:
                        #print(symstream[label_changes[idx],0] - symstream[label_changes[idx+1],0])
                        middle_idx = label_changes[idx] + math.ceil((label_changes[idx+1] - label_changes[idx])/2)
                        symstream[label_changes[idx]:middle_idx, label_col_idx]     = symstream[label_changes[idx]-1, label_col_idx]
                        symstream[middle_idx:label_changes[idx+1]+1, label_col_idx] = symstream[label_changes[idx+1]+1, label_col_idx]

        if "extend" in self.config.options:
            for label_col_idx in self.config.label_columns:
                label_changes = np.where(np.diff(symstream[:,label_col_idx], axis=0) != 0)[0]
                labels_to_change = self.config.options['extend']['affected_labels']
                front  = self.config.options['extend']['extension_period_s'][0]
                back   = self.config.options['extend']['extension_period_s'][1]

                for idx in label_changes:
                    cur_label = symstream[idx  , label_col_idx]
                    nxt_label = symstream[idx+1, label_col_idx]

                    if cur_label in labels_to_change: #and nxt_label not in labels_to_change:
                        # Extend cur_label by back - always do this and also overwrite front extension
                        extension_idx = symstream.shape[0]
                        time_idx      = np.where(symstream[idx:,0] >= symstream[idx,0] + back)[0]
                        if len(time_idx) > 0:
                            extension_idx = min([extension_idx, time_idx[0] + idx])
                        next_valid = np.where(symstream[idx+1:,label_col_idx] != nxt_label)[0]
                        if next_valid.size > 0:
                            extension_idx = min([extension_idx, next_valid[0] + idx +1])
                        symstream[idx:extension_idx, label_col_idx] = cur_label

                    if cur_label not in labels_to_change and nxt_label in labels_to_change:
                        # Extend nxt_label by front only if label before is not to be extended
                        extension_idx = 0
                        time_idx      = np.where(symstream[:idx,0] <= symstream[idx,0] - front)[0]
                        if len(time_idx) > 0:
                            extension_idx = max([extension_idx, time_idx[-1]])
                        last_valid = np.where(symstream[:idx+1,label_col_idx] != cur_label)[0]
                        if last_valid.size > 0:
                            extension_idx = max([extension_idx, last_valid[-1]])
                        symstream[extension_idx:idx+1, label_col_idx] = nxt_label

        if "remove_labels" in self.config.options:
            for label in self.config.options["remove_labels"]:
                for label_col_idx in self.config.label_columns:
                    sampling_period = np.diff(symstream[:,0]).mean()
                    symstream = symstream[symstream[:,label_col_idx] != label['int'],:]
                    discontinuities = np.where(np.diff(symstream[:,0]) > (sampling_period * 1.5))[0]
                    symstream[:,0] = np.linspace(0, (symstream.shape[0]-1)* sampling_period, symstream.shape[0])
                    for idx in range(len(discontinuities)):
                        if(len(symstream[discontinuities[idx]+1:,0]) > 0):
                            symstream[discontinuities[idx]+1:,1:-2] = self.augment(symstream[discontinuities[idx]+1:,0]-symstream[discontinuities[idx],0],
                                                                                   symstream[discontinuities[idx]+1:,1:-2],
                                                                                   symstream[discontinuities[idx],1:-2])

        X_cols = list(range(1,symstream.shape[1]))
        for ycol in self.config.label_columns:
            X_cols.remove(ycol)
        return np.hstack((
                            symstream[:,0].reshape((-1,1)),
                            symstream[:,X_cols].reshape((-1,len(X_cols))),
                            symstream[:,self.config.label_columns].reshape((-1,len(self.config.label_columns)))
                        ))

    @abc.abstractmethod
    def augment(self, *args, **kwargs):
        """This method defines how the transitions are treated if parts have been removed due to the option remove_unknown"""
        pass

class ThermalLabelRefactoring(LabelRefactoring, Layer, layer=Layer.Type.Line):
    def _build_parameter_datastruct(self, all_params):
        params = dict()
        for key in all_params:
            if all_params[key] != np.nan and all_params[key] > 0:
                params[key] = all_params[key]
        return params

    @property
    def beta_z(self):
        return self._build_parameter_datastruct(self.config.environments_apps_zones[self.config.env]['snk']['zone_config'].beta_z)

    @property
    def beta_z_heat(self):
        return self._build_parameter_datastruct(self.config.environments_apps_zones[self.config.env]['snk']['zone_config'].beta_z_heat)

    @property
    def beta_z_cool(self):
        return self._build_parameter_datastruct(self.config.environments_apps_zones[self.config.env]['snk']['zone_config'].beta_z_cool)

    @property
    def T_idle(self):
        return self._build_parameter_datastruct(self.config.environments_apps_zones[self.config.env]['snk']['zone_config'].T_idle)

    @property
    def np_beta_z(self):
        beta_z = []
        for col_name in self.config.beta_z_names:
            beta_z.append(self.beta_z[col_name])
        return beta_z

    @property
    def np_T_idle(self):
        T_idle = []
        for col_name in self.config.beta_z_names:
            T_idle.append(self.T_idle[col_name])
        return T_idle

    def _thermal_function(self, T, beta, t):
        return T * np.exp(-beta * t)

    def augment(self, *args, **kwargs):
        time = args[0]
        T_z  = args[1]
        T0_z = args[2]
        if 'beta_key' in kwargs:
          betas = getattr(self, kwargs['beta_key'], self.beta_z) # TODO dirty
        else:
          betas = self.beta_z

        if len(T0_z.shape) == 1:
            T0_z = T0_z.reshape((1, T0_z.shape[0]))
        T_aug = np.empty(T_z.shape)
        for col_idx in range(T_z.shape[1]):
            beta_val = betas[self.config.beta_z_names[col_idx]]
            T_aug[:,col_idx] = T_z[:,col_idx] + self._thermal_function((T0_z[0, col_idx] - T_z[0,col_idx]), beta_val, time)
        return T_aug

