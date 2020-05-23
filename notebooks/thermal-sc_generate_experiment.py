import os
if os.getcwd().split('/')[1] == 'home':
  base_path = "/home/miedlp/Documents/workspace/toolkit/"
  DATA_TASK_FILES = False
  PROC_TASK_FILES = False
elif os.getcwd().split('/')[1] == 'local':
  base_path = "/local/scratch/toolkit/"
  DATA_TASK_FILES = False
  PROC_TASK_FILES = False
else:
  base_path = "/itet-stor/miedlp/net_scratch/toolkit/"
  DATA_TASK_FILES = True
  PROC_TASK_FILES = True

os.environ['EXOT_ACCESS_DIR'] = base_path + "benchmark_platforms_access/"
os.chdir(base_path + 'datapro/')
os.getcwd()

import toml    as TOML              # TOML library to handle config files
import pandas  as pd                # Pandas for data manipulation
import seaborn as sb                # Statistical data visualisation
import pathlib as PATH              # Used to convert strings to path objects for (de-)serialisation
import types   as tp                # Library used for online method development
from shutil import copyfile

# scikit-learn packets
import sklearn.base                 #
import sklearn.svm                  # LinearSVC, SVC
import sklearn.pipeline             # make_pipeline
import sklearn.preprocessing        # StandardScaler
import sklearn.decomposition        # PCA
import sklearn.naive_bayes          # GaussianNB
import sklearn.metrics              # CM

# ExOT packets
import exot                       # Dataprocessing
import exot.experiment            # Experiment for execution
import exot.util                  # General utilities
import exot.layer                 # Data manipulation layers
from exot.util       import *
from exot.layer      import *
from exot.channel    import *
from exot.experiment import *
from exot.util.attributedict import AttributeDict
from exot.util.logging import get_root_logger


def generate_and_write_experiment(config):
  channel    = ChannelFactory()(config["EXPERIMENT"]["channel"])
  experiment = ExperimentFactory()(config["EXPERIMENT"]["type"], config=config, channel=channel)

  experiment.generate()
  experiment.print_duration()
  #experiment.channel.bootstrap_analyses()
  experiment.write()

for configfile in [
                 "./configurations/thermal-sc_benchmarks.toml",
                 "./configurations/thermal-sc_dragonboard_imitation.toml",
                 "./configurations/thermal-sc_augmentation_parameters.toml",
                ]:
  print("Generating experiment specified in", configfile)
  config     = TOML.load(configfile)
  generate_and_write_experiment(config)

def generate_task_file(data_spec_name, proc_spec_name):
  os.chdir(base_path + 'datapro/notebooks')
  if DATA_TASK_FILES:
    data_arton_file = "repetitouch_" + data_spec_name + ".local.sh"
    copyfile("template_data.arton.sh", data_arton_file)
    os.system(f"sed -i 's/TBD/{data_spec_name}/' {data_arton_file}")
    os.system(f"chmod +x {data_arton_file}")

  if PROC_TASK_FILES:
    proc_arton_file = "repetitouch_" + proc_spec_name + ".local.sh"
    copyfile("template_proc.arton.sh", proc_arton_file)
    os.system(f"sed -i 's/TBD/{proc_spec_name}/' {proc_arton_file}")
    os.system(f"chmod +x {proc_arton_file}")
  os.chdir(base_path + 'datapro/')

for configfile in [
                   "./configurations/thermal-sc_repetitouch.toml",
                  ]:

  print("Generating experiment specified in", configfile)
  config     = TOML.load(configfile)

  # Configfile bootstrap
  base_templates          = dict()
  base_templates["proc"]  = AttributeDict(config["ANALYSES"]["proc_template"])
  base_templates["data"]  = AttributeDict(config["ANALYSES"]["data_template"])
  del config["ANALYSES"]["proc_template"]
  del config["ANALYSES"]["data_template"]

  envs = list(base_templates["data"]['DATASET']['var_env'])
  del base_templates["data"]['DATASET']['var_env']

  dimensions = dict(base_templates["data"]["DATASET"]["matcher_data"]["var_dimensions"])
  del base_templates["data"]["DATASET"]["matcher_data"]["var_dimensions"]

  analysis_templates = dict()
  to_delete          = list()
  for analysis_name in config["ANALYSES"]:
    if 'template' in analysis_name:
      template_name = analysis_name.split('template_')[-1]
      analysis_templates[template_name] = AttributeDict(config["ANALYSES"][analysis_name])
      to_delete.append(analysis_name)
      print("Found template %s / %s" % (analysis_name, template_name))
  for analysis_name in to_delete:
    if 'template' in analysis_name:
      del config["ANALYSES"][analysis_name]

  for name, spec in analysis_templates.items():
    if name == "binary":
      apps = AttributeDict(spec["DATASET"]["label_mapping"])
    else:
      apps = [None]
    for app in apps:
      for env in envs:
        # ------------------------------ Data Generation Spec --------------------------------------
        if app is not None:
          new_data_spec_name = ("data_" + env + "_" + name + "_" + apps[app]['str']).lower()
        else:
          new_data_spec_name = ("data_" + env + "_" + name).lower()

        new_data_spec = AttributeDict(base_templates["data"])
        if hasattr(spec, "DATASET"):
          for key, value in spec["DATASET"].items():
            if type(spec["DATASET"][key]) in [dict, AttributeDict]:
              new_data_spec["DATASET"][key] = AttributeDict(spec["DATASET"][key])
            else:
              new_data_spec["DATASET"][key] = spec["DATASET"][key]

        if app is not None:
          new_data_spec["DATASET"]["label_mapping"]              = {app:apps[app]}
          if "runs" in apps[app].keys():
            new_data_spec["DATASET"]["sample_selection"]["runs"] = list(apps[app]["runs"])

        new_data_spec['DATASET']['env']                        = env
        new_data_spec["DATASET"]["matcher_data"]["dimensions"] = dimensions[env]

        config["ANALYSES"][new_data_spec_name]                 = new_data_spec

        # ------------------------------ Data Processing Spec --------------------------------------
        for model_type in ['CNNLSTM']: #['LSTM', 'CNNLSTM']:
          for num_layers in [4]: #[1,2,3,4,5,6,7,8]:
            for num_hidden in [128]: #[4,8,11,12,16,32,64,128,256,384,512,640,768,896]:
              if app is not None:
                new_proc_spec_name = ("proc_" + env + "_" + name + "_" + apps[app]['str'] + "_" + str(num_layers) + "_" + str(num_hidden) + '_' + model_type).lower()
              else:
                new_proc_spec_name = ("proc_" + env + "_" + name + "_" + str(num_layers) + "_" + str(num_hidden) + '_' + model_type).lower()

              new_proc_spec = AttributeDict(base_templates["proc"])
              new_proc_spec["DATASET"]["load_from"] = new_data_spec_name

              new_proc_spec["MODEL"]["num_layers"]  = num_layers
              new_proc_spec["MODEL"]["num_hidden"]  = num_hidden
              new_proc_spec["MODEL"]["type"]        = model_type

              config["ANALYSES"][new_proc_spec_name] = new_proc_spec

              generate_task_file(new_data_spec_name, new_proc_spec_name)
              print("Added new analyses " + new_data_spec_name + " and " + new_proc_spec_name)

  generate_and_write_experiment(config)

