# Orca TF Estimator

This is an example to demonstrate how to use Analytics-Zoo's Orca TF Estimator API to run distributed
Tensorflow and Keras on Spark.

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip).

```bash
conda create -y -n analytics-zoo python==3.7.7
conda activate analytics-zoo
pip install analytics_zoo-${VERSION}-${TIMESTAMP}-py2.py3-none-${OS}_x86_64.whl
pip install tensorflow==1.15.0
pip install psutil
```

Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Run examples on local

```bash
python lenet_mnist_graph.py --cluster_mode local 
```

```bash
python lenet_mnist_keras.py --cluster_mode local
```

## Run examples on yarn cluster
```bash
python lenet_mnist_graph.py --cluster_mode yarn
```

```bash
python lenet_mnist_keras.py --cluster_mode yarn
```

## Additional Resources
The application is also be able to run on spark standalone cluster or in yarn cluster mode.
Please refer to the following links to learn more details.

1. [Orca Overview](https://analytics-zoo.github.io/master/#Orca/overview/) and [Orca Context](https://analytics-zoo.github.io/master/#Orca/context/)
2. [Download and install Analytics Zoo](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)

