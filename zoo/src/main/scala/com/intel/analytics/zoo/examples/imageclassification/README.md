## Image Classification example
This example illustrates how to do the image classification with pre-trained model

### Run steps
1. Prepare pre-trained models

Download pre-trained models from [Image Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/image-classification.md)

2. Prepare predict dataset

Put your image data for prediction in one folder.

3. Run the example

```bash
master=... // spark master

modelPath=... // model path

imagePath=... // image path

ANALYTICS_ZOO_HOME=...
ZOO_JAR_PATH=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-VERSION-jar-with-dependencies.jar

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
