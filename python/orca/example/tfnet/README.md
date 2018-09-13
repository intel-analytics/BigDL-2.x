## TFNet Object Detection example

TFNet can encapsulate a frozen TensorFlow graph as an Analytics Zoo layer for inference.

This example illustrates how to use a pre-trained TensorFlow object detection model
to make inferences using Analytics Zoo on Spark.

## Model and Data Preparation
1. Prepare a pre-trained TensorFlow object detection model. You can download from [tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

In this example, we use `frozen_inference_graph.pb` of the `ssd_mobilenet_v1_coco` model downloaded from [here](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz).

2. Prepare the image dataset for inference. Put the images to do prediction in the same folder.


## Run this example
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:
```bash
SPARK_HOME=the root directory of Spark
MASTER=...
ANALYTICS_ZOO_ROOT=the root directory of the Analytics Zoo project
ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
ANALYTICS_ZOO_PY_ZIP=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-python-api.zip
ANALYTICS_ZOO_JAR=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar
ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
PYTHONPATH=${ANALYTICS_ZOO_PY_ZIP}:$PYTHONPATH


${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PY_ZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tfnet/predict.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tfnet/predict.py \
    --image path_to_image_folder \
    --model path_to_tensorflow_graph
```

__Options:__
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path of the TensorFlow object detection model. Local file system, HDFS and Amazon S3 are supported.
* `--partition_num` The number of partitions to cut the dataset into. Default is 4.

## Results
The result of this example will be the detection boxes (y_min, x_min, y_max, x_max) of the input images, with the first detection box of an image having the highest prediction score.

We print the detection box with the highest score of the first prediction result to the console.
