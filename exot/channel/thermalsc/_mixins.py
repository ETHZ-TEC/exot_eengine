import abc

from exot.channel.mixins.mldataset import GenericDataSetHandler
from exot.util.logging import get_root_logger

__all__ = ("ThermalAppDecectionDataSetHandler")

class ThermalAppDecectionDataSetHandler(GenericDataSetHandler):
    # -------------------------------------------------------------------------------------------- #
    #                                           Overwrites                                         #
    # -------------------------------------------------------------------------------------------- #
    def dataset_parameters_to_dict(self) -> dict:
        param_dict = GenericDataSetHandler.dataset_parameters_to_dict(self)
        param_dict['unknown_label']        = self.unknown_label
        param_dict['num_labels']           = self.num_labels
        param_dict['batch_size_timesteps'] = self.batch_size_timesteps
        return param_dict

    @property
    def unknown_label(self):
        return getattr(self, '_unknown_label', False)

    @unknown_label.setter
    def unknown_label(self, value: bool) -> None:
        setattr(self, '_unknown_label', value)


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
        return getattr(self, '_num_labels', 0)

    @num_labels.setter
    def num_labels(self, value: int) -> None:
        self._num_labels = value

