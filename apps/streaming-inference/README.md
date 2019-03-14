# Streaming Inferece Examples
Streaming inference/predict based on [analytics-zoo](https://github.com/intel-analytics/analytics-zoo)

## Summary
Quick example about integrating analytics-zoo inference/predict service into streaming related applications. We prepared 2 examples: text classification and object detection. Most code are based on [Spark Streaming](https://spark.apache.org/docs/2.2.0/streaming-programming-guide.html)/[Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) examples.

## Environment
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Datasets and pre-trained models

1. [Streaming Object Detection]()
Using pre-trained SSD model to detect objects in images.

2. [Streaming Text Classification]()
Using pre-trained LSTM/CNN model to classify network input string.

