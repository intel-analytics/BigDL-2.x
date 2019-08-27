## OpenVINO ResNet_v1_50 example
This example illustrates how to use a pre-trained OpenVINO optimized model to make inferences with OpenVINO toolkit as backend using Analytics Zoo.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## PrepareOpenVINOResNet
TensorFlow models cannot be directly loaded by OpenVINO. It should be converted to OpenVINO optimized model and int8 optimized model first. You can use PrepareOpenVINOResNet or [OpenVINO toolkit](https://docs.openvinotoolkit.org/2018_R5/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html) to finish this job. Herein, we focused on PrepareOpenVINOResNet.

Download [TensorFlow ResNet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [validation image set](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino/val_bmp_32.tar) and [OpenCVLibs](https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/openvino/opencv_4.0.0_ubuntu_lib.tar). Extract files from these packages. 

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export ANALYTICS_ZOO_JAR=export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

MODEL_PATH=dir of resetNet 50 checkpoint, i.e., resnet_v1_50.ckpt
VALIDATION=dir of validation images and val.txt, i.e., val_bmp_32
OPENCVLIBS=dir of OpenCV libs

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
    com.intel.analytics.zoo.examples.vnni.openvino.PrepareOpenVINOResNet \
    -m ${MODEL_PATH} -v ${VALIDATION} -l ${OPENCVLIBS}
```

__Options:__
- `-m` `--model`: The dir of resetNet 50 checkpoint.
- `-b` `--batchSize`: The batch size of input data. Default is 4.
- `-l` `--openCVLibs`: The number of iterations to run the performance test. Default is 1.
- `-v` `--validationFilePath`: dir of validation images and val.txt.
- `--subset`: Number of images in validation file path. Note that it should be align with val.txt.


__Sample Result files in MODEL_PATH__:
```
resnet_v1_50.ckpt
resnet_v1_50_inference_graph.bin
resnet_v1_50_inference_graph-calibrated.bin
resnet_v1_50_inference_graph-calibrated.xml
resnet_v1_50_inference_graph.mapping
resnet_v1_50_inference_graph.xml
```

Amount them, `resnet_v1_50_inference_graph.xml` and `resnet_v1_50_inference_graph.bin` are OpenVINO optimized ResNet_v1_50 model and weight, `resnet_v1_50_inference_graph-calibrated.xml` and `resnet_v1_50_inference_graph-calibrated.bin` are OpenVINO int8 optimized ResNet_v1_50 model and weight. Both of them can be loaded by OpenVINO or Zoo.

## Options
* `--image` The path where the images are stored. It can be either a folder or an image path. Local file system, HDFS and Amazon S3 are supported.
* `--model` The path to the TensorFlow object detection model.
* `--partition_num` The number of partitions.

## Results
We print the inference result of each batch.
