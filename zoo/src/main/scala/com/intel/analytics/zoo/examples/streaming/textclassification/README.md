# Analytics Zoo Streaming Text Classification
Based on Streaming example NetworkWordCount and Zoo text classification example. Network inputs (Strings) are pre-processed and classified by zoo. We applied a simple text classification model based on zoo example.

## Environment
* Apache Spark (This version needs to be same with the version you use to build Analytics Zoo)
* [Analytics Zoo](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/)

## Datasets and pre-trained models
* Get Pre-trained model & word index from [Text Classification Example](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/docs/ProgrammingGuide/text-classification.md).

## Run this example
Make sure all nodes can access pre-trained model and word index.

### Method One

1. TERMINAL 1: Running Netcat
```
nc -lk [port]
```

2. TERMINAL 2: Start StreamingTextClassification
```
MASTER=...
model=... // model path. Local file system/HDFS/Amazon S3 are supported
indexPath=... // word index path. Local file system/HDFS/Amazon S3 are supported
port=... // The same port with nc command
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    --class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingTextClassification \
    --model ${model} --indexPath ${indexPath} --port ${port}
```

3. TERMINAL 1: Input text in Netcat
```
hello world
It's a fine day
```

## Method Two

1. TERMINAL 2: Start StreamingTextClassification
```
MASTER=...
model=... // model path. Local file system/HDFS/Amazon S3 are supported
indexPath=... // word index path. Local file system/HDFS/Amazon S3 are supported
inputFile=... // dir to the file you will input data
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 5g \
    --class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingTextClassification \
    --model ${model} --indexPath ${indexPath} --inputFile ${inputFile}
```

2. TERMINAL 1: Input text using echo
```
export inputFile=... // dir to the file you will input data
echo "hello world" > ${inputFile}/s  // s can be any new file
```

**Then, you can see output in TERMINAL 2.**
```
Probability distributions of top-5:
alt.atheism 0.009092145
sci.crypt 0.01049567
rec.motorcycles 0.010525138
talk.politics.mideast 0.011516748
rec.sport.hockey 0.014714912
```


## Better Performance with Inference Model
[Inference Model](https://analytics-zoo.github.io/0.4.0/#ProgrammingGuide/inference/#inference-model) is a thread-safe package in Analytics Zoo aiming to provide high level APIs to speed-up development.

To enable this feature, simply replace `--class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingTextClassification` with `--class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingInferenceTextClassification` in Step 2.
