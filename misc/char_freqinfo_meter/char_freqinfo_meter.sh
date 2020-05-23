#!/bin/bash
# This script can be used to control the device operating frequency using the acpi interface.
# The service has a fixed sampling rate of 30ms.
# The binaries have to be placed into the same directory as the script, and will save the measurements to the steps_userspace.csv/
# The service saves the measurements to ~/.local/share/freqinfo-meter/
#
#SERVICE=0    # Uncomment if the meter as a service should be used, otherwise it will be started via the binary

T=30 # ms
NUM_SAMPLES_PER_LEVEL=500

# ------------------------------------------------------------------------------------------------ #
#                                              T440p                                               #
#NUM_CORES=8
#NUM_FREQS=15
#CORE_FREQUENCIES=(800MHz 900MHz 1000MHz 1.10GHz 1.30GHz 1.40GHz 1.50GHz 1.60GHz 1.70GHz 1.80GHz 1.90GHz 2.10GHz 2.20GHz 2.30GHz 2.40GHz)
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
#                                              T460s                                               #
NUM_CORES=3
NUM_FREQS=15
CORE_FREQUENCIES=(400MHz 600MHz 700MHz 800MHz 1.10GHz 1.30GHz 1.40GHz 1.60GHz 1.80GHz 1.90GHz 2.10GHz 2.30GHz 2.50GHz 2.60GHz 2.70GHz)
# ------------------------------------------------------------------------------------------------ #

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                            Do not change anything below this line.                               #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
cmd_meter="$(pwd)/dbg_freqinfo-meter -t $((T*1000)) --asap -l $(pwd)/steps_userspace_vll.csv"
cmd_meter="$(pwd)/rel_freqinfo-meter -t $((T*1000)) --asap -l $(pwd)/steps_userspace_vll.csv"

if [ -n "$SERVICE" ]; then
  T=30
  systemctl start --user freqinfo-meter
else
  ${cmd_meter}&
  pid=$!
fi

# Scale frequency from lowest to higest.
sudo bash -c 'for i in {0..'$(expr ${NUM_CORES} - 1)'}; do cpufreq-set -g userspace -c $i; done'
sudo bash -c 'for i in {0..'$(expr ${NUM_CORES} - 1)'}; do cpufreq-set -f '${CORE_FREQUENCIES[0]}' -c $i; done'
for i in {0..14}; do
  echo "Evaluating ${CORE_FREQUENCIES[i]}..."
  sudo bash -c 'for i in {0..'$(expr ${NUM_CORES} - 1)'}; do cpufreq-set -g userspace -c $i; done'
  sudo bash -c 'for i in {0..'$(expr ${NUM_CORES} - 1)'}; do cpufreq-set -f '${CORE_FREQUENCIES[i]}' -c $i; done'
  echo "Take ${NUM_SAMPLES_PER_LEVEL} Samples"
  sleep $((${NUM_SAMPLES_PER_LEVEL} * ${T} / 1000))
done

if [ -n "$SERVICE" ]; then
  systemctl stop --user freqinfo-meter
else
  sudo kill ${pid}
fi

echo "DONE - killed " 
echo ${pid}
# ------------------------------------------------------------------------------------------------ #

