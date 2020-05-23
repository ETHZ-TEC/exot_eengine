#!/bin/bash
#
#SBATCH  --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/miedlp/net_scratch/toolkit/datapro/data/_logs/repetitouch_TBD_%j.out      # where to store the output ( %j is the JOBID )
#SBATCH  --cpus-per-task=1                   # Use 1 CPUS
#SBATCH  --gres=gpu:1                        # Use 1 GPUS
#SBATCH  --mem=32G                           # use 32GB
CUDA_HOME=/itet-stor/miedlp/net_scratch/cuda-10.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64

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
