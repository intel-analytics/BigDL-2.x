# Analytics Zoo Streaming Text Classification
Based on Streaming example NetworkWordCount and Zoo text classification example. Network inputs (Strings) are pre-processed and classified by zoo. We applied a simple text classification model based on zoo example.

## Environment
* Python (2.7, 3.5 or 3.6)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo ([install analytics-zoo]((https://analytics-zoo.github.io/master/#PythonUserGuide/install/) ) via __pip__ or __download the prebuilt package__.)

## Datasets and pre-trained models
* Pre-trained model & word index: Save trained text classification model and word index in [Text Classification](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/text-classification.md).

## Run this example
Make sure all nodes can access model and word index.

1. TERMINAL 1: Running Netcat
```
nc -lk [port]
```

2. TERMINAL 2: Start streaming_text_classification
```
MASTER=...
model=... // model path. Local file system/HDFS/Amazon S3 are supported
indexPath=... // word index path. Local file system/HDFS/Amazon S3 are supported
port=... // The same port with nc command
${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    streaming_text_classification.py \
    --model ${model} --indexPath ${indexPath} --port ${port}
```

3. TERMINAL 1: Input text in Netcat
```
hello world
It's a fine day
```

Then, you can see output in TERMINAL 2.
```
Predicts:
0.84158087 soc.religion.christian
0.08710906 talk.religion.misc
0.06145671 alt.atheism
0.0035352204 talk.politics.guns
9.059753E-4 rec.sport.baseball
```
