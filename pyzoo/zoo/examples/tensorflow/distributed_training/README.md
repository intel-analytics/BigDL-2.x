# Distributed Tensorflow on Spark/BigDL

This is an example to demonstrate how to use Analytics-Zoo API to run distributed
Tensorflow on Spark/BigDL.

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and
the `research/slim` directory to `PYTHONPATH`.

## Run the Training Example

```shell
export ANALYTICS_ZOO_HOME=... # Please edit this accordingly
export SPARK_HOME=... # Please edit this accordingly

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] train_lenet.py
```

## Run the Evaluation Example

```shell
export ANALYTICS_ZOO_HOME=... # Please edit this accordingly
export SPARK_HOME=... # Please edit this accordingly

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] evaluate_lenet.py
```

