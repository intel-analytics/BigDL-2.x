#!/bin/bash

export ZOO_HOME=$ZOO_HOME
export ZOO_HOME_DIST=$ZOO_HOME/dist
export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export ZOO_JAR=`find ${ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ZOO_PY_ZIP=`find ${ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*python-api.zip"`
export PYTHON_API_ZIP_PATH=$ZOO_PY_ZIP
export ZOO_JAR_PATH=$ZOO_JAR
export ZOO_CONF=${ZOO_HOME_DIST}/conf/spark-bigdl.conf
export PYTHONPATH=${ZOO_PY_ZIP}:$PYTHONPATH
export FTP_URI=$FTP_URI

echo "#1 start example test for textclassification"
if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
	echo "analytics-zoo-data/data/glove.6B.zip already exists" 
else
	wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data 
	unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi 
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then 
	echo "analytics-zoo-data/data/20news-18828.tar.gz already exists" 
else
	wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P ana lytics-zoo-data/data 
	tar zxf analytics-zoo-data/data/20news-18828.tar.gz -d analytics-zoo-data/data/20news-18828
fi
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 20g \
    --py-files ${PYTHON_API_ZIP_PATH},${ZOO_HOME}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --jars ${ZOO_JAR_PATH} \
    --conf spark.driver.extraClassPath=${ZOO_JAR_PATH} \
    --conf spark.executor.extraClassPath=${ZOO_JAR_PATH} \
    ${ZOO_HOME}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 2 \
    --data_path analytics-zoo-data/data