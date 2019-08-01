# OpenVINO

## Basic Requirements
1. Python & TensorFlow
2. OpenVINO 2018 installed

## Basic workflow of OpenVINO
Basic workflow of OpenVINO optimization and calibration.

```
(TensorFlow) -optimization-> IR model -calibration-> int8 IR model
```

Note that OpenVINO `int8` IR promises higher performance than IR model (FP32) by sacrificing a few precision (1% by default).

Both IR and int8 IR can be loaded by OpenVINO, and make prediction. User should ensure that optimization and calibration are correctly performed.

## Auto Calibration
Assume you have OpenVINO IR model (converted from Caffe or TensorFlow), and you want to convert it into `int8` model. To achive that, you can use OpenVINO `calibration_tool` or use our `auto_calibration` tool, which provides simpler API based on `calibration_tool`.

**Basic Usage**
```bash
python auto_calibration.py -m {OpenVINO IR model, *.xml} -i {validation data}
```

**API usage**
```bash
python auto_calibration.py -h
```

**Validation Data**
The key of calibration is a carefully prepared validation data. For different models, you should prepare different validation data. Currently, our `auto_calibration` tool only support image classification and object detection models and validation data.

**1. Image Classification**
Two kinds of validation dir are supported:
- Image in label dir 
```bash
+-- 0
|   +-- Image_001.jpg
|   +-- Image_003.jpg
+-- 1
|   +-- Image_002.jpg
|   +-- Image_004.jpg
```
- val.txt with images
```
+-- val.txt
+-- Image_001.jpg
+-- Image_002.jpg
+-- Image_003.jpg
+-- Image_004.jpg
```
val.txt contains image path and label, separated by space.
```
Image_001.jpg 0
Image_002.jpg 1
Image_003.jpg 0
Image_004.jpg 1
```

**2. Object Detection**
For OpenVINO 2018, only VOC format is supported for Object Detection. We simplify the dir structure:
```
+-- classes.txt
+-- Image_001.jpg
+-- Image_002.jpg
+-- Image_003.jpg
+-- Image_004.jpg
+-- Image_001.xml
+-- Image_002.xml
+-- Image_003.xml
+-- Image_004.xml
```

Herein `*.xml` is annotation files for images. `classes.txt` is the labels. Example of `classes.txt` is shown below:
```
none_of_the_above 0
aeroplane 1
bicycle 2
bird 3
boat 4
bottle 5
bus 6
car 7
cat 8
chair 9
cow 10
diningtable 11
dog 12
horse 13
motorbike 14
person 15
pottedplant 16
sheep 17
sofa 18
train 19
tvmonitor 20
```

## References
1. [OpenVINO](https://software.intel.com/en-us/openvino-toolkit) 
2. [Analytics-Zoo](https://github.com/intel-analytics/analytics-zoo)
3. [TensorFlow](https://www.tensorflow.org/)

