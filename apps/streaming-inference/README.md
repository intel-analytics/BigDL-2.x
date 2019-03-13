# Zoo_Stream_Inference
Streaming inference/predict based on [analytics-zoo](https://github.com/intel-analytics/analytics-zoo)

## Streaming Inferece Example for Zoo
Quick example about integrating analytics-zoo inference/predict service into streaming related applications. We prepared 2 examples: text classification and object detection. Most code are based on [Spark Streaming](https://spark.apache.org/docs/2.2.0/streaming-programming-guide.html)/[Structured Streaming](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html) examples.

### Text Classification
Based on Streaming example NetworkWordCount. The difference is that network inputs are pre-processed and classified by zoo. We applied a simple text classification model in zoo example.


### Object Detection
**1. Local case**
Imagining we have image files in local dir, and we want to detect objects in these images. In streaming case, it's not an easy task to read image files into stream without help of Kafka etc. So, we first package image pathes into text files, then read text file in streaming example. After reading image path, we read image content and make predict (Object Detection based on SSD).

**2. Distrubted case [Working in Progress]**
In this case, we need a third part framework that can package images into binary streaming, such as Kafka.
