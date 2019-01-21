# Distributed Tensorflow on Spark/BigDL

This is an example to demonstrate how to use Analytics-Zoo API to run distributed
Tensorflow and Keras on Spark/BigDL.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model Preparation

In this example, we will use the **slim** library to construct the model. You can
clone it [here](https://github.com/tensorflow/models/tree/master/research/slim) and
the `research/slim` directory to `PYTHONPATH`.

## Run the Training Example after pip install

```bash
python train_lenet.py
```

For Keras users:

```bash
python train_mnist_keras.py
```

## Run the Training Example with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] train_lenet.py
```

For Keras users:

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] train_mnist_keras.py
```

## Run the Evaluation Example after pip install

```bash
python evaluate_lenet.py
```

For Keras users:

```bash
python evaluate_mnist_keras.py
```

## Run the Evaluation Example with prebuilt package

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] evaluate_lenet.py
```

For Keras users:

```bash
export ANALYTICS_ZOO_HOME=... # the directory where you extract the downloaded Analytics Zoo zip package
export SPARK_HOME=... # the root directory of Spark

sh $ANALYTICS_ZOO_HOME/bin/spark-submit-with-zoo.sh --master local[4] evaluate_mnist_keras.py
```


