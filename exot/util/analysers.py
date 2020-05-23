"""Misc helpers"""
import numpy as np
import pandas as pd
from exot.util.attributedict import LabelMapping

# --------------------------------------------------------------------------------------------------
def all_labels_in_dataset(experiment, sort_str=False, **kwargs):
    for key in ['phase', 'labelcolumn', 'io']:
        if key not in kwargs.keys():
            raise ValueError(f"key {key} not specified in kwargs!")
    ingest_args = kwargs.copy()
    labels = np.array([])
    #for rep in range(1):
    for rep in range(experiment.config.EXPERIMENT.PHASES[ingest_args['phase']].repetitions):
        ingest_args['io']['rep'] = rep
        #for cur_run in [experiment.phases[ingest_args['phase']]['antutu2']]:
        for cur_run in experiment.phases[ingest_args['phase']].values():
            try:
                cur_run.ingest(**ingest_args)
                labels = np.concatenate((labels, cur_run.i_rawstream[ingest_args['labelcolumn']].unique()))
            except:
                print("Could not ingest run", cur_run)

    labels = pd.DataFrame(labels)[0].unique().flatten()
    if sort_str:
        return _sort_str_labels(labels)
    else:
        return labels

# --------------------------------------------------------------------------------------------------
def sort_str_labels(labels):
    convert = lambda x: str(x)
    return np.array(list(map(convert, labels)))

# --------------------------------------------------------------------------------------------------
def generate_unique_labels_mapping(labels_keys, labels_str):
    if labels_str is None:
        for key in labels_keys:
            if key not in labels_str.keys():
                labels_str[key] = key
        labels_map = dict([(y,{'int':x, 'str':str(labels_keys[x])}) for x,y in enumerate(set(labels_keys))])
    else:
        labels_map = dict([(y,{'int':x, 'str':labels_str[x]}) for x,y in enumerate(set(labels_keys))])
    return LabelMapping(labels_map)

