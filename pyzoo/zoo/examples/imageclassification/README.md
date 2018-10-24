# Image Classification example
This example illustrates how to do the image classification with pre-trained model

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Run steps
### Prepare pre-trained models
Download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

### Prepare predict dataset
Put your image data for prediction in one folder.

### Run the example
Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
modelPath=... // model path
imagePath=... // image path
topN=... // top n prediction
partitionNum=... // A suggestion value of the minimal partition number
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --jars ${ZOO_JAR_PATH} \
    --conf spark.driver.extraClassPath=${ZOO_JAR_PATH} \
    --conf spark.executor.extraClassPath=${ZOO_JAR_PATH} \
    path/to/predict.py -f $imagePath --model $modelPath --topN 5 --partition_num ${partitionNum}
```
See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification#options) for more configurable options for this example.
