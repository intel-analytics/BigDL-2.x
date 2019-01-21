## OpenVINO Object Detection example

This example illustrates how to use a pre-trained TensorFlow object detection model
to make inferences with OpenVINO toolkit as backend using Analytics Zoo, which delivers a significant boost for inference speed ([up to 19.9x](https://software.intel.com/en-us/blogs/2018/05/15/accelerate-computer-vision-from-edge-to-cloud-with-openvino-toolkit)).

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model and Data Preparation
1. Download the following files for pre-trained TensorFlow `faster_rcnn_resnet101_coco`: [model](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino/TF_faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb), 
[pipeline configure](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino/TF_faster_rcnn_resnet101_coco_2018_01_28/pipeline.config) and
[extensions configure](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino/TF_faster_rcnn_resnet101_coco_2018_01_28/faster_rcnn_support.json)

2. Prepare the image dataset for inference. Put the images to do prediction in the same folder.


## Run this example after pip install
```bash
image_path=directory path containing images
model_path=path to frozen_inference_graph.pb
pipeline_path=path to pipeline.config
extensions_path=path to faster_rcnn_support.json

python predict.py --image ${image_path} --model ${model_path} --pipeline ${pipeline_path} --extensions ${extensions_path}
```

See [here](#options) for more configurable options for this example.


## Run this example with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the directory where you extract the downloaded Analytics Zoo zip package
MASTER=local[*]
image_path=directory path containing images
model_path=path to frozen_inference_graph.pb
pipeline_path=path to pipeline.config
extensions_path=path to faster_rcnn_support.json

${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master $MASTER \
    predict.py \
    --image ${image_path} \
    --model ${model_path} \
    --pipeline ${pipeline_path} \
    --extensions ${extensions_path}
```

See [here](#options) for more configurable options for this example.


## Options
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path to the TensorFlow object detection model.
* `--pipeline` The path to the pipeline configure file.
* `--extensions` The path to the extensions configure file.

## Results
We print the detection result of the first image to the console.
