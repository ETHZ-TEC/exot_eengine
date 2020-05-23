#!/bin/bash
#
#SBATCH  --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/miedlp/net_scratch/toolkit/datapro/data/_logs/repetitouch_TBD_%j.out      # where to store the output ( %j is the JOBID )
#SBATCH  --cpus-per-task=4                   # Use 4 CPUS
#SBATCH  --mem=32G                           # use 32GB
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate toolkit
#
echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
python -u thermal-sc_execute_analysis.arton.py Thermal-SC_Repetitouch TBD
echo finished at: `date`
exit 0;
