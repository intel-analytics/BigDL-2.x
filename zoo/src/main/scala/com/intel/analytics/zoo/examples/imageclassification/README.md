## Image Classification example
This example illustrates how to do the image classification with pre-trained model

### Run steps
1. Prepare pre-trained models

Download pre-trained models from https://github.com/intel-analytics/zoo/tree/master/docs/models/imageclassification

2. Prepare predict dataset

Put your image data for prediction in the ./image folder.

3. Run the example

```bash
master=... // spark master

modelPath=... // model path

imagePath=... // image path


spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.imageclassification.Predict \
./zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f $imagePath --model $modelPath --partition 4 --topN 5
```