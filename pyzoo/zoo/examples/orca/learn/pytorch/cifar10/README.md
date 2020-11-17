# Pytorch Cifar10 example
We demostrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Analytics Zoo. We use a simple convolutional nueral network model to train on Cifar10 dataset, which is a dataset for image classification. See [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) for the original single-node version of this example provided by Pytorch.

## Prepare environments

We recommend you to use Anaconda to prepare the environments, especially if you want to run on a yarn cluster

```
conda create -n zoo python=3.7 #zoo is conda environment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install torch
pip install torchvision
pip install ray==0.8.4
```

# Run on local after pip install

```
python cifar10_pytorch.py
```

See [here](#Options) for more configurable options for this example.

# Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=path to your hadoop conf directory
python cifar10_pytorch.py --cluster_mode yarn-client
```

See [here](#Options) for more configurable options for this example.

# Options

* `--cores` The number of cores you want to use on each node. Default is 4.

* `--num_nodes` The number of nodes to be used in the cluster. Default is 1.
  

* `--workers_per_node` The number of workers to run on each node. Default is 2.


* `--memory` The memory you want to use on each node. Default is 10g.

* `--epochs` The number of epochs to train. Default is 2.

* `--batch_size` The worker batch size for training per executor. Default is 16.

# Results

You can find the results of training and validation

```
 Train stats: [{'num_samples': 50000, 'epoch': 1, 'batch_count': 3125, 'train_loss': 2.0466214977264405, 'last_train_loss': 2.0413246154785156}, {'num_samples': 50000, 'epoch': 2, 'batch_count': 3125, 'train_loss': 1.58512735704422, 'last_train_loss': 2.135589361190796}]
 
 Validation stats: {'num_samples': 10000, 'batch_count': 313, 'val_loss': 1.4752447093963623, 'last_val_loss': 0.5625}
```

