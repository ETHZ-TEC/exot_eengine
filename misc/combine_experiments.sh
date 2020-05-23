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

