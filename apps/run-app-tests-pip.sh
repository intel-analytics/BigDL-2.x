#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh


echo "#9 start app test for image-augmentation"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time9=$((now-start))
echo "#9 image-augmentation time used:$time9 seconds"

echo "#11 start app test for image-augementation-3d"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augementation-3d

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time11=$((now-start))
echo "#11 image-augementation-3d time used:$time11 seconds"


