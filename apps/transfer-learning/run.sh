export SPARK_HOME=/home/xiaxue/spark/spark-2.1.1-bin-hadoop2.6
export ANALYTICS_ZOO_HOME=/home/xiaxue/529/analytics-zoo/dist/
export MASTER=local[1]

${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 8g
