# Image Classification example
This example illustrates how to do the image classification with pre-trained model

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via pip or download the prebuilt package.

## Prepare pre-trained models
Download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

## Prepare predict dataset
Put your image data for prediction in one folder.

## Run after pip install
```bash
export SPARK_DRIVER_MEMORY=10g
modelPath=... // model path
imagePath=... // image path
topN=... // top n prediction
partitionNum=... // A suggestion value of the minimal partition number
python predict.py -f $imagePath --model $modelPath --topN 5 --partition_num ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
modelPath=... // model path
imagePath=... // image path
topN=... // top n prediction
partitionNum=... // A suggestion value of the minimal partition number
${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    path/to/predict.py -f $imagePath --model $modelPath --topN 5 --partition_num ${partitionNum}
```
See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.
