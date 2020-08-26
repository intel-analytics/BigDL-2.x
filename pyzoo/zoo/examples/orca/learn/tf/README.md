# Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed
Tensorflow and Keras on Spark.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and add
the `research/slim` directory to `PYTHONPATH`.

```bash

git clone https://github.com/tensorflow/models/

export PYTHONPATH=$PWD/models/research/slim:$PYTHONPATH
```


## Run Keras model example after pip install

```bash
export MASTER=local[4]
python keras_lenet.py
```

## Run Keras model example with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] keras_lenet.py
```

## Run tf graph model example after pip install

```bash
export MASTER=local[4]
export SPARK_DRIVER_MEMORY=2g
python graph_lenet.py
```
## Run tf graph model example with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

bash $ANALYTICS_ZOO_HOME/bin/spark-submit-python-with-zoo.sh --master local[4] graph_lenet.py
```