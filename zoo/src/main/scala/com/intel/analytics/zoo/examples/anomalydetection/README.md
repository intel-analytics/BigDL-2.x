# Overview

This is the Scala example for anomaly detection. It demonstrates how to use Analytics Zoo to build anomaly detector based on LSTM.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Data Prepraration: 
We use one of the datasets in Numenta Anomaly Benchmark (NAB[https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv]) for demo, i.e. NYC taxi passengers dataset, which contains 10320 records, each indicating the total number of taxi passengers in NYC at specific time. 
Before you run the example, download the data, unzip it and put into a directory.

The following scripts we provide will serve to download and extract the data for you:
```bash
bash ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh dir
```
Remarks:
- `ANALYTICS_ZOO_HOME` is the folder where you extract the downloaded package and `dir` is the directory you wish to locate the corresponding downloaded data.
- If `dir` is not specified, the data will be downloaded to the current working directory.

## Run the anomaly detection example

``` bash
   export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
   nyc_path=the directory containing containing NBA nyc_taxi.csv data
   master=... // spark master
   ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
   --master $master \
   --driver-memory 4g \
   --executor-memory 4g \
   --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection \
   --inputDir ${nyc_path}
```
