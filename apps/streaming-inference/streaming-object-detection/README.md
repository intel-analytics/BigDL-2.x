# Streaming Object Detection
Imagining we have image files in file system, and we want to detect objects in these images. In streaming case, it's not an easy task to read image files with help of a third part framework (such HDFS or Kafka). To simplify this example, we package image pathes into text files, which can be read by Spark Streaming or Structured Streaming API. Then, these image pathes will be passed to executors through RDD. After getting image pathes (RDD) from streaming API, executor will read image content from file systems, and make prediction. The predicted results (images with boxes) will be stored to output dir.

We will have two applications in this example, ImagePathWriter and StreamingObjectDetection. ImagePathWriter will package image pathes into text files, and StreamingObjectDetection will make prediction based on text files and image files.

## 1. Local case
ImagePathWriter read local image files, and write text files into local file system. On the other hand, StreamingObjectDetection will read text files, and feed image bytes into pre-tranined model. The predicted result will be write to local file sytem.

## 2. Distrubted case
In this example, we need a third part framework (such HDFS or S3). To simplify this example, we choose HDFS. Similar to local case, ImagePathWriter packages pathes into text files while StreamingObjectDetection makes prediction based on image files. The only difference is that files are read and write to HDFS.