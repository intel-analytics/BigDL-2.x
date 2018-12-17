# Overview

This is the Scala example for anomaly detection. We demostrate how to use Analytics Zoo to build anomaly detector based on LSTM.

## Data: 
   We use one of the datasets in Numenta Anomaly Benchmark (NAB[https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv]) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at specific time. 
   Before you run the example, download the data, unzip it and put into ./data/NAB/nyc_taxi/.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Run the anomaly detection example

``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection \
   --inputDir ./data/NAB/nyc_taxi/
```
