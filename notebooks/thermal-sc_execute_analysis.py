import sys
if len(sys.argv) != 4:
    raise Exception("This script takes exactly three arguments, the experiment and the analysis name, e.g.: python -u thermal-sc_execute_analysis.local.py Thermal-SC_Repetitouch DataWhiteBoxBilbo $(pwd)")

import os
base_path = sys.argv[3] + "/../../"
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

path_to_serialised_experiment = PATH.Path("data/" + str(sys.argv[1]) + "/_experiment.pickle")
experiment = AppExecExperiment.read(path_to_serialised_experiment, diff_and_replace=False)

experiment.channel.bootstrap_analysis(str(sys.argv[2]))
experiment.channel.analyses[str(sys.argv[2])].execute()
experiment.channel.analyses[str(sys.argv[2])].write()
copyfile(list(experiment.save_path.joinpath('_logs').glob("*" + os.environ['SLURM_JOB_ID'] + "*"))[0], experiment.channel.analyses[str(sys.argv[2])].path.joinpath("debug_" + os.environ['SLURM_JOB_ID'] + ".out"))

