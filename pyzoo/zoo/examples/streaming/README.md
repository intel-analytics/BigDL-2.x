# Analytics Zoo Streaming Inference Examples
Streaming inference/predict based on [analytics-zoo](https://github.com/intel-analytics/analytics-zoo)

## Summary
Quick example about integrating analytics-zoo inference/predict service into streaming related applications. We prepared 2 examples:

1. [Streaming Object Detection](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/streaming/objectdetection) uses pre-trained SSD model to detect objects in images.
2. [Streaming Text Classification](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/streaming/textclassification) uses pre-trained CNN/LSTM model to classify text from network input.

## Environment
* Python 3.6/3.7
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo ([install analytics-zoo]((https://analytics-zoo.github.io/master/#PythonUserGuide/install/) ) via __pip__ or __download the prebuilt package__.)

## Datasets and pre-trained models
**1. Streaming Object Detection**
* Datasets: [COCO](http://cocodataset.org/#home)
* Pre-trained model: [SSD 300x300](https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model)

**2. Streaming Text Classification**
* Pre-trained model: Save trained text classification model and word2index in [Text Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/text-classification.md).
