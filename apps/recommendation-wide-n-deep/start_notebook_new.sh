export ANALYTICS_ZOO_HOME=/Users/guoqiong/intelWork/git/analytics-zoo/dist_2.1
export SPARK_HOME=/Users/guoqiong/intelWork/tools/spark/spark-2.1.1-bin-hadoop2.7
#export SPARK_HOME=/Users/guoqiong/intelWork/tools/spark/spark-1.6.0-bin-hadoop2.6
MASTER=local[*]
bash ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g \
