# Transformer-Attention is all you need
Transformer is a neural network architect proposed in the paper "Attention Is All You Need"
(https://arxiv.org/abs/1706.03762). The network architecture is based solely on attention
mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine
translation tasks show the model to be superior in quality while being more parallelizable
and requiring significantly less time to train.

In Multi-head Attention Sentiment Analyisi notebook, we use keras API build the core layer: 
Multi-head Attention layer and apply it to a sentiment analysis example.


## Environment
* Python 2.7/3.5/3.6
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Run with Jupyter
* Install Zoo following the doc: https://analytics-zoo.github.io/0.2.0/#PythonUserGuide/install/#install-without-pip
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master local[4] \
    --driver-memory 12g
```
