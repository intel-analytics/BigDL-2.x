# Analytics Zoo Streaming Object Detection
Imagining we have pre-trained model and image files in file system, and we want to detect objects in these images. In streaming case, it's not an easy task to read image files with help of a third part framework (such as HDFS or Kafka). To simplify this example, we package image paths into text files. Then, these image paths will be passed to executors through streaming API. Executors will read image content from file systems, and make prediction. The predicted results (images with boxes) will be stored to output dir.

So, there are two applications in this example: image_path_writer and streaming_object_detection. image_path_writer will package image paths into text files. Meanwhile, streaming_object_detection read image path from those text files, then read image content and make prediction.

## Environment
* Python (2.7, 3.5 or 3.6)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo ([install analytics-zoo]((https://analytics-zoo.github.io/master/#PythonUserGuide/install/) ) via __pip__ or __download the prebuilt package__.)

## Datasets and pre-trained models
* Datasets: [COCO](http://cocodataset.org/#home)
* Pre-trained model: [SSD 300x300](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)

## Run this example
Make sure all nodes can access image files, model and text files. Local file system/HDFS/Amazon S3 are supported.

Pls ensure all paths exist and accessible, and `streaming_path` is empty. Note that `streaming_object_detection` and `image_path_writer` should use the same `streaming_path`.

1. Start streaming_object_detection
```
MASTER=...
model=... // model path. Local file system/HDFS/Amazon S3 are supported
streaming_path=... // text files location. Only local file system is supported
output_path=... // output path of prediction result. Only local file system is supported
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 5g \
    --executor-memory 5g \
    streaming_object_detection.py \
    --streaming_path ${streaming_path} --model ${model} --output_path ${output_path}
```

2. Start image_path_writer
```
MASTER=...
img_path=... // image path. Only local file system is supported
streaming_path=... // text files. Only local file system is supported
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 5g \
    image_path_writer.py \
    --streaming_path ${streaming_path} --img_path ${img_path}
```

## Results
Images with objects boxes will be save to ${output} dir.
