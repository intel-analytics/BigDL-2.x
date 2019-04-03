# Analytics Zoo Streaming Object Detection
Imagining we have pre-trained model and image files in file system, and we want to detect objects in these images. In streaming case, it's not an easy task to read image files with help of a third part framework (such as HDFS or Kafka). To simplify this example, we package image paths into text files. Then, these image paths will be passed to executors through streaming API. Executors will read image content from file systems, and make prediction. The predicted results (images with boxes) will be stored to output dir.

So, there are two applications in this example: ImagePathWriter and StreamingObjectDetection. ImagePathWriter will package image paths into text files. Meanwhile, StreamingObjectDetection read image path from those text files, then read image content and make prediction.

## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/)

## Datasets and pre-trained models
* Datasets: [COCO](http://cocodataset.org/#home)
* Pre-trained model: [SSD 300x300](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)

## Run this example
Make sure all nodes can access image files, model and text files. Local file system/HDFS/Amazon S3 are supported.

1. Start StreamingObjectDetection
```
MASTER=...
model=... // model path. Local file system/HDFS/Amazon S3 are supported
streamingPath=... // text files location. Local file system/HDFS/Amazon S3 are supported
output=... // output path of prediction result. Local file system/HDFS/Amazon S3 are supported
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    --class com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection \
    --streamingPath ${streamingPath} --model ${model} --output ${output}
```

2. Start ImagePathWriter
```
MASTER=...
imageSourcePath=... // image path. Local file system/HDFS/Amazon S3 are supported
streamingPath=... // text files. Local file system/HDFS/Amazon S3 are supported
${SPARK_HOME}/bin/spark-submit-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    --class com.intel.analytics.zoo.examples.streaming.objectdetection.ImagePathWriter \
    --streamingPath ${streamingPath} --imageSourcePath ${imageSourcePath}
```
## Results
Images with objects boxes will be save to ${output} dir.