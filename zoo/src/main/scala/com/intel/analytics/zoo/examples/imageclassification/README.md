## Image Classification example
This example illustrates how to do the image classification with pre-trained model

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Prepare pre-trained models
Download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

## Prepare predict dataset
Put your image data for prediction in one folder.

## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=200g
master=... // spark master

modelPath=... // model path

imagePath=... // image path

ANALYTICS_ZOO_HOME=...
ZOO_JAR_PATH=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar 
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.imageclassification.Predict \
${ZOO_JAR_PATH} -f $imagePath --model $modelPath --partition 4 --topN 5
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.