# Multi-head Attention Sentiment Analysis Example
This example introduces the Transformer network architecture based solely on attention mechanisms.

The idea comes from [Attention is All You Need](https://arxiv.org/abs/1706.03762), an influential paper with a catchy title that brings innovative change in the field of machine translation. This paper demonstrated how high performance can be achieved without convolutional or recurrent neural networks, which were previously regarded as the go-to architecture for machine translation.

In this example, we show how to resolve the sentimental analysis task with IMDB data with Multi-head Attention network architecture.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

NOTE: Please install `keras2` in your environment as we uses keras dataset api to download and process training data. `keras2` can be installed easily via pip.

## Run with prebuilt package
We recommend to run this example in a cluster instead of local mode to get better performance. Also please set stack size to a large number to avoid StackOverflow exception:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
export VENV_HOME=the parent directory of venv.zip and venv folder

PYSPARK_DRIVER_PYTHON=${VENV_HOME}/venv/bin/python PYSPARK_PYTHON=venv.zip/venv/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.executor.extraJavaOptions="-Xss512m" \
    --conf spark.driver.extraJavaOptions="-Xss512m" \
    --master yarn-client \
    --executor-cores 8 \
    --num-executors 10 \
    --driver-memory 20g \
    --executor-memory 100g \
    --archives ${VENV_HOME}/venv.zip \
    transformer.py
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#for-yarn-cluster) for more running guidance for running in yarn cluster.

## Results
You can find the accuracy information from the log during the training process:
```
INFO  DistriOptimizer$:395 - [Epoch 1 25120/25000][Iteration 157][Wall Clock 1168.885167551s] Trained 160 records in 7.518009571 seconds. Throughput is 21.282228 records/second. Loss is 0.32141894.
INFO  DistriOptimizer$:439 - [Epoch 1 25120/25000][Iteration 157][Wall Clock 1168.885167551s] Epoch finished. Wall clock time is 1169003.884269 ms
Train finished.
Evaluating...
Evaluated result: 0.803879976273, total_num: 25000, method: Top1Accuracy
```
