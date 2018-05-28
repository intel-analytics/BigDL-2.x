## TFNet Object Detection example

TFNet can encapsulate a tensorflow freezed graph as Analytics-Zoo layer for inference.

This example illustrates how to use tensorflow pretrained object detection model
to make inferences on Spark.

### Run steps
1. Prepare pre-trained models

Download pre-trained models from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

In this example, we will use the ssd_mobilenet_v1_coco model.

2. Prepare predict dataset

Put your image data for prediction in the ./image folder.

3. Run the example

```bash
master=... // spark master

modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported

imagePath=... // image path. Local file system are supported.

outputPath=... // output path. Currently only support local file system.

classNamePath=... // the path of coco_classname.txt file

spark-submit \
--verbose \
--master $master \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
./zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar --image $imagePath --output $outputPath --model $modelPath --partition 4
```

## Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
