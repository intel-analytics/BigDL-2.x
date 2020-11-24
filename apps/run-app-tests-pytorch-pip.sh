#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

RUN_PART1=0
RUN_PART2=0
RUN_PART3=0
RUN_PART4=0
if [ $1 = 1 ]; then
	RUN_PART1=1
	RUN_PART2=0
	RUN_PART3=0
	RUN_PART4=0
elif [ $1 = 2 ]; then
	RUN_PART1=0
	RUN_PART2=1
	RUN_PART3=0
	RUN_PART4=0
elif [ $1 = 3 ]; then
	RUN_PART1=0
	RUN_PART2=0
	RUN_PART3=1
	RUN_PART4=0
elif [ $1 = 4 ]; then
	RUN_PART1=0
	RUN_PART2=0
	RUN_PART3=0
	RUN_PART4=1
else
	RUN_PART1=1
	RUN_PART2=1
	RUN_PART3=1
	RUN_PART4=1
fi

if [ $RUN_PART3 = 1 ]; then
echo "#15 start app test for pytorch face-generation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
sed -i '/plt./d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
python ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "pytorch face-generation failed"
    exit $exit_status
fi
now=$(date "+%s")
time15=$((now-start))
echo "#15 pytorch face-generation time used:$time15 seconds"
fi