## Object Detection Python example
This example illustrates how to detect objects given image and model
### Run steps
1. Build 

```bash
./build.sh 
```


2. Setup environment

Add python zip file to `PYTHONPATH`

```bash
export DL_PYTHON_HOME=${path_to_analytics-zoo}/models/target/bigdl-models-0.1-SNAPSHOT-python-api.zip

export PYTHONPATH=$PYTHONPATH:${path_to_bigdl}/dist/conf/spark-bigdl.conf:$DL_PYTHON_HOME
```

3. Run the example

```bash
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
```