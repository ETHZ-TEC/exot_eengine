base_path='../data/_logs'
old_path=$(pwd)
cd ${base_path}

#for file in $(ls -d benchmarks_*); do 
#  tmp=$(echo ../Thermal-SC_Benchmarks/${file/repetitouch_/} | sed 's/\(.*\)_/\1\/debug_/')
#  mv ${file} ${tmp}
#done

#for file in $(ls -d dragonboard_imitation_*); do 
#  tmp=$(echo ../Thermal-SC_DragonboardImitation/${file/repetitouch_/} | sed 's/\(.*\)_/\1\/debug_/')
#  mv ${file} ${tmp}
#done

for file in $(ls -d repetitouch_*); do
  tmp=$(echo ../Thermal-SC_Repetitouch/${file/repetitouch_/} | sed 's/\(.*\)_/\1\/debug_/')
  mv ${file} ${tmp}
done

#for file in $(ls -d final_*); do
#  tmp=$(echo ../Thermal-SC_Final/${file/repetitouch_/} | sed 's/\(.*\)_/\1\/debug_/')
#  mv ${file} ${tmp}
#done

cd ${old_path}
