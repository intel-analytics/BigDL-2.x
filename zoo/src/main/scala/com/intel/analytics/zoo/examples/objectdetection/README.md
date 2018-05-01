## Object Detection example
This example illustrates how to detect objects in image with pre-trained model

### Run steps
1. Prepare pre-trained models

Download pre-trained models from https://github.com/intel-analytics/zoo/tree/master/docs/models/objectdetection

2. Prepare predict dataset

Put your image data for prediction in the ./image folder.

3. Run the example

```bash
master=... // spark master

modelPath=... // model path

imagePath=... // image path

outputPath=... // output path

spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.objectdetection.Predict \
./zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar --image $imagePath --output $outputPath --model $modelPath --partition 4
```

## Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
