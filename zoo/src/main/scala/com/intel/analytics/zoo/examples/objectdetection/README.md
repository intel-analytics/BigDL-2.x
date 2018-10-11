## Object Detection example
This example illustrates how to detect objects in image with pre-trained model

### Run steps
#### Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

#### Prepare pre-trained models

Download pre-trained models from [Object Detection](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/object-detection.md)

#### Prepare predict dataset

Put your image data for prediction in the ./image folder.

#### Run the example

```bash
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

master=... // spark master

modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported

imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster.

outputPath=... // output path. Currently only support local file system.

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--verbose \
--master $master \
--driver-memory 200g \
--executor-memory 200g \
--class com.intel.analytics.zoo.examples.objectdetection.Predict \
--image $imagePath --output $outputPath --modelPath $modelPath --partition 4
```

### Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]
