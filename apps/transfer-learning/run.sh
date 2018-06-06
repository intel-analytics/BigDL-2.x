export SPARK_HOME=/home/arda/chentao/apache/spark-2.2.0-bin-hadoop2.6
export ANALYTICS_ZOO_HOME=/home/arda/chentao/work/zoo/dist
export MASTER=local[4]

${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 20g  \
    --total-executor-cores 1  \
    --executor-cores 1  \
    --executor-memory 20g
