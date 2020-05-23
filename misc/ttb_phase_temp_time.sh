#!/bin/bash
USER_HOME=/home/rocketlogger
BASE_PATH=${USER_HOME}/data/

RUNNAME=`basename $(ls -d ${BASE_PATH}/*/*/)`

IFS='_' read -ra PARSED <<< "$RUNNAME"
IFS='-' read -ra PARAMS <<< "${PARSED[1]}"

echo "${#PARAMS[@]}"
echo "$(expr ${#PARAMS[@]} / 2)"
for ((iter=0;iter<$(expr ${#PARAMS[@]} / 2);iter++));
do
  temperature=${PARAMS[$(expr ${iter} \* 2 + 0)]}
  wait_time=${PARAMS[$(expr ${iter} \* 2 + 1)]}
  echo "$(date +"%Y-%m-%d %T") -- ttb set ${temperature}"
  ${USER_HOME}/bin/ttb set ${temperature}
  echo $(${USER_HOME}/bin/ttb sense out)
  echo "$(date +"%Y-%m-%d %T") -- ttb on"
  ${USER_HOME}/bin/ttb on
  echo "$(date +"%Y-%m-%d %T") -- sleep ${wait_time}"
  sleep ${wait_time}
  echo $(${USER_HOME}/bin/ttb sense out)
done
echo "$(date +"%Y-%m-%d %T") -- ttb off"
${USER_HOME}/bin/ttb off
echo "DONE"
sleep 100

