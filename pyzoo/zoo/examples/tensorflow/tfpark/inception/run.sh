export ANALYTICS_ZOO_HOME=/path/to/analytics-zoo
DATA_PATH=hdfs://path/to/imagenet
EXECUTOR_CORES=24
TOTAL_EXECUTOR_CORES=192
SPARK_MASTER=yarn


bash ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
 --master $SPARK_MASTER \
 --executor-cores $EXECUTOR_CORES \
 --total-executor-cores $TOTAL_EXECUTOR_CORES \
 --executor-memory 175G \
 --driver-memory 20G \
 --conf spark.network.timeout=10000000 \
 inception.py \
 --batchSize 1536 \
 --learningRate 0.0896 \
 -f $DATA_PATH \
 --maxIteration 62000
