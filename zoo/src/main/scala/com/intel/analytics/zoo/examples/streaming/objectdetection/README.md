# Streaming Object Detection
Imagining we have pre-trained model and image files in file system, and we want to detect objects in these images. In streaming case, it's not an easy task to read image files with help of a third part framework (such HDFS or Kafka). To simplify this example, we package image pathes into text files. Then, these image pathes will be passed to executors through streaming API. Executors will read image content from file systems, and make prediction. The predicted results (images with boxes) will be stored to output dir.

So, there are two applications in this example: ImagePathWriter and StreamingObjectDetection. ImagePathWriter will package image pathes into text files. Meanwhile, StreamingObjectDetection read image path from thoses text file, then read image content and make prediction.

## Environment
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Datasets and pre-trained models

* Datasets: [COCO](http://cocodataset.org/#home)
* Pre-trained model: [SSD 300x300](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)

## Run this example
Make sure all nodes can access image files, model and text files. Local file system/HDFS/Amazon S3 are supported.

1. Start StreamingObjectDetection
```
MASTER=...
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported
streamingPath=... // text files. Local file system/HDFS/Amazon S3 are supported
output=... // output path of prediction result
${SPARK_HOME}/bin/spark-submit --master local[*] --driver-memory 10g --class com.intel.analytics.zoo.apps.streaming.StreamingObjectDetection target/streaming-object-detection-0.1.0-SNAPSHOT-jar-with-dependencies.jar --modelPath [modelPath] --streamingPath [streamingPath] --output [output]
```

2. Start ImagePathWriter
```
MASTER=...
imagePath=... // image path. Local file system/HDFS/Amazon S3 are supported
streamingPath=... // text files. Local file system/HDFS/Amazon S3 are supported
${SPARK_HOME}/bin/spark-submit --class com.intel.analytics.zoo.apps.streaming.ImagePathWriter target/streaming-object-detection-0.1.0-SNAPSHOT-jar-with-dependencies.jar --streamingPath [streamingPath] --imageSourcePath [imagePath]
```
