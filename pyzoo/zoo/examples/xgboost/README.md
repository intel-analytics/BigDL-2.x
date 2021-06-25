# XGBoost Example

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Download data
1. If demo mode is specified, no download is needed.
2. If using other dataset, store the dataset in the csv file. Sample dataset (Boston Housing dataset) can be downloaded from here [boston](http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv)

## Run with Prebuilt package
```
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
```

## Options
* '--file-path' or '-f', where data is stored, default will be current folder (Required Argument).
* '--demo' or '-d', whether to use demo data or not. 
