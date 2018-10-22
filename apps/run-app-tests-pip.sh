#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

echo "#1 start app test for anomaly-detection"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi
chmod +x ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
sed "s/nb_epoch=20/nb_epoch=2/g; s/batch_size=1024/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py > ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "anomaly-detection-nyc-taxi time used:$time1 seconds"

echo "#9 start app test for image-augmentation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-memory 1g  \
        --executor-memory 1g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py
now=$(date "+%s")
time9=$((now-start))
echo "#9 image-augmentation time used:$time9 seconds"

echo "#11 start app test for image-augementation-3d"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augementation-3d
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --driver-memory 1g  \
        --executor-memory 1g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augementation-3d.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augementation-3d.py
now=$(date "+%s")
time11=$((now-start))
echo "#11 image-augementation-3d time used:$time11 seconds"

# This should be done at the very end after all tests finish.
clear_up
