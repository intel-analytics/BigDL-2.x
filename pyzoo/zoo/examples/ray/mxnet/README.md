# MXNet LeNet example

Here we demonstrate how to easily run synchronous distributed [MXNet](https://github.com/apache/incubator-mxnet) training using MXNetTrainer implemented on top of [RayOnSpark](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/)
We use the LeNet model to train on MNIST dataset for handwritten digit recognition. 
See [here](https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html) for the original single-node version of this example.


## Prepare environments
Follow steps 1 to 4 [here](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/#steps-to-run-rayonspark) 
to prepare your python environment.

You also need to install **MXNet** in your conda environment via pip. We have tested on MXNet 1.6.0.
```bash
pip install mxnet==1.6.0
```
If you are running on Intel Xeon scalable processors, you probably want to install the [MKLDNN](https://github.com/oneapi-src/oneDNN) version of MXNet for better performance:
```bash
pip install mxnet-mkl==1.6.0
```

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install)
for more running guidance after pip install. 

## Run on local after pip install
```
python lenet_mnist.py -n 2
```
See [here](#Options) for more configurable options for this example.

## Run on yarn cluster for yarn-client mode after pip install 
```
python lenet_mnist.py --hadoop_conf ...# path to your hadoop/yarn directory --conda_name ...# your conda name
```
 
See [here](#Options) for more configurable options for this example.

## Options
- `--num_workers` The number of workers. Default is 4.

**Options for yarn only**
- `--hadoop_conf` This option is **required** when you want to run on yarn. The path to your configuration folder of hadoop.
- `--conda_name` This option is **required** when you want to run on yarn. Your conda environment's name.
- `--executor_cores` The number of slave(executor)'s cpu cores you want to use. Default is 8.
- `--executor_memory` The size of slave(executor)'s memory you want to use. Default is 10g.
- `--driver_memory` The size of driver's memory you want to use. Default is 2g.

