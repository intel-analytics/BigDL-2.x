## Image Classification example
This example illustrates how to do the image classification with pre-trained model

### Run steps
1. Prepare pre-trained models

Download pre-trained models from https://github.com/intel-analytics/zoo/tree/master/docs/models/imageclassification

2. Prepare predict dataset

Put your image data for prediction in one folder.

3. Run the example

```bash
modelPath=... // model path

imagePath=... // image path

topN=... // top n prediction

ANALYTICS_ZOO_HOME=
PYTHON_API_ZIP_PATH=${ANALYTICS_ZOO_HOME}/lib/zoo-VERSION-SNAPSHOT-python-api.zip
ZOO_JAR_PATH=${ANALYTICS_ZOO_HOME}/lib/zoo-VERSION-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

spark-submit \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --jars ${ZOO_JAR_PATH} \
    --conf spark.driver.extraClassPath=${ZOO_JAR_PATH} \
    --conf spark.executor.extraClassPath=${ZOO_JAR_PATH} \
    path/to/predict.py -f $imagePath --model $modelPath --topN 5
```