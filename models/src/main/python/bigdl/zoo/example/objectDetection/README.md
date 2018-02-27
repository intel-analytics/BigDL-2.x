## Object Detection Python example
This example illustrates how to detect objects given image and model
### Run steps
1. Build 

./build.sh 

2. Setup environment

Add model jar to `BIGDL_CLASSPATH` and python zip file to `PYTHONPATH`

export DL_PYTHON_HOME=${path_to_analytics-zoo}/models/target/bigdl-models-0.1-SNAPSHOT-python-api.zip

export PYTHONPATH=$PYTHONPATH:${path_to_bigdl}/dist/conf/spark-bigdl.conf:$DL_PYTHON_HOME

export BIGDL_CLASSPATH=$BIGDL_CLASSPATH:${path_to_analytics-zoo}/models/target/models-0.1-SNAPSHOT-jar-with-dependencies.jar

3. Run the example

master=... // spark master

modelPath=... // model path

imagePath=... // image path

outputPath=... // output path

${SPARK_HOME}/bin/spark-submit \
        --master $master \
        --driver-cores 4  \
        --driver-memory 10g  \
        --total-executor-cores 8  \
        --executor-cores 4  \
        --executor-memory 4g \
        --py-files ${DL_PYTHON_HOME},Predict.py  \
        --properties-file ${BIGDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${path_to_analytics-zoo}/models/target/models-0.1-SNAPSHOT-jar-with-dependencies.jar \
         Predict.py ${modelPath} ${imagePath} ${outputPath}
