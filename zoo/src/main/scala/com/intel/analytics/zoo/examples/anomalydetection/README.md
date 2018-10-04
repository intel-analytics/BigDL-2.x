# Overview

This is the Scala example for anomaly detection. We demostrate how to use Analytics Zoo to build anomaly detector based on LSTM.

## Data: 
   We used one of the dataset in Numenta Anomaly Benchmark (NAB[https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv]) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at a corresonponding time spot. 
   Before you run the example, download the data, unzip it and put into ./data/ml-1m/.

## Run the anomaly detection example
    Command to run the example in Spark local mode:
```
    spark-submit \
    --master local[physcial_core_number] \
    --driver-memory 10g --executor-memory 20g \
    --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetectionExample \
    ./dist/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar \
    --inputDir ./data/ml-1m \

```

    Command to run the example in Spark yarn mode:
```
    spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-cores 8 \
    --num-executors 4 \
    --driver-memory 10g \
    --executor-memory 150g \
    --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetectionExample \
    ./dist/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar \
    --inputDir hdfs://xxx

```
