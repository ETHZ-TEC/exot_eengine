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
base_path='/local/scratch/toolkit/datapro/data/'
#experiment='Thermal-SC_Repetitouch'
experiment='Thermal-SC_Repetitouch_Visual_Inspection'
phase='bm'
envs=('Frodo' 'Bilbo')
filenames=( 'snk.config.json' 'snk.debug.txt' 'snk.log.csv' 'src.config.json' )


for env in "${envs[@]}"; do 
  for usecase_full in $(ls -d ${base_path}/${experiment}/${phase}*); do
    usecase=$(basename ${usecase_full})
    for filename in "${filenames[@]}"; do 
      cnt=0
      for file in $(ls -d ${base_path}/${experiment}*/${usecase}/${env}/*_${filename}); do
      #for file in $(ls -d ${base_path}/${experiment}*${env}*/${usecase}/${env}/*_${filename}); do
        mkdir -p ${base_path}/${experiment}/${usecase}/${env}/
        cp ${file} ${base_path}/${experiment}/${usecase}/${env}/$(seq -f "%03g" ${cnt} ${cnt})_${filename}
        let "cnt+=1"
      done
    done
    echo "${env} - ${usecase}: $(ls -l ${base_path}/${experiment}/${usecase}/${env}/*snk*.csv | egrep -c '^-') samples"
  done
done

