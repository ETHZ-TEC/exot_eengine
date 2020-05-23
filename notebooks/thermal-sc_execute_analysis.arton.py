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
import sys
if len(sys.argv) != 3:
    raise Exception("This script takes exactly two arguments, the experiment and the analysis name, e.g.: python -u thermal-sc_execute_analysis.local.py Thermal-SC_Repetitouch DataWhiteBoxBilbo")

import os
base_path = "/itet-stor/miedlp/net_scratch/toolkit/"
os.environ['EXOT_ACCESS_DIR'] = base_path + "benchmark_platforms_access/"
os.chdir(base_path + 'datapro/')
os.getcwd()

import toml    as TOML              # TOML library to handle config files
import pandas  as pd                # Pandas for data manipulation
import seaborn as sb                # Statistical data visualisation
import pathlib as PATH              # Used to convert strings to path objects for (de-)serialisation
import types   as tp                # Library used for online method development
from shutil import copyfile

#from copy   import copy, deepcopy   # TODO
#from pprint import pprint           # TODO

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

