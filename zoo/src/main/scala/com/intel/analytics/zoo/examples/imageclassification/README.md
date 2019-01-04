## Image Classification example
This example illustrates how to do the image classification with pre-trained model

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.
## Run the example
### Download the pre-trained model
You can download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

### Prepare predict dataset
Put your image data for prediction in one folder.

### Run this example
Run the following command for Spark local mode (MASTER=local[*]) or cluster mode:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported
imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster and please use file:///... for local files.
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
     --master ${MASTER} \
     --driver-memory 8g \
     --executor-memory 8g \
     --verbose \
     --conf spark.executor.cores=1 \
     --total-executor-cores 4 \           
     --class com.intel.analytics.zoo.examples.imageclassification.Predict \
     -f $imagePath --model $modelPath --partition 4 --topN 5
```
