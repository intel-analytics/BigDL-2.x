# Streaming Text Classification
Based on Streaming example NetworkWordCount and Zoo text classification example. Network inputs (Strings) are pre-processed and classified by zoo. We applied a simple text classification model based on zoo example.

## Environment
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Datasets and pre-trained models

* Word embeding file: [golve](https://nlp.stanford.edu/projects/glove/)
* Pre-trained model & word index: Save trained text classification model and word index in [Text Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/text-classification.md).

## Run this example
Make sure all nodes can access model, word index and glove.

1. TERMINAL 1: Running Netcat
```
nc -lk [port]
```

2. TERMINAL 2: Start StreamingTextClassification
```
MASTER=...
embeddingPath=... // glove path. Local file system/HDFS/Amazon S3 are supported
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported
indexPath=... // word index path. Local file system/HDFS/Amazon S3 are supported
${SPARK_HOME}/bin/spark-submit --master local[*] --class com.intel.analytics.zoo.apps.streaming.StreamingTextClassification ../streaming-text-classification-0.1.0-SNAPSHOT-jar-with-dependencies.jar --embeddingPath /root/workspace/data/glove.6B --model /root/workspace/model/textclassification.model  --indexPath  /root/workspace/model/textclassification.txt
```

3. TERMINAL 1: Input text in Netcat
```
hello world
It's a fine day
```
Then, you can see output in TERMINAL 2.
