# Pytorch Tensorboard example
We demostrate how to easily show the graphical results of runing synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Analytics Zoo. We use a simple convolutional nueral network model to train on fashion-MNIST dataset. See [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) for the original single-node version of this example provided by Pytorch.

## Prepare environments

We recommend you to use Anaconda to prepare the environments, especially if you want to run on a yarn cluster

```
conda create -n zoo python=3.7 #zoo is conda environment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install torch
pip install torchvision
pip install ray==0.8.4
pip install tensorboardX
```

# Run on local after pip install

```
python tensorboard.py
tensorboard --logdir=runs
```

Then open `https://localhost:6006` to see the result figures.

See [here](#Options) for more configurable options for this example.

# Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=path to your hadoop conf directory
python cifar10_pytorch.py --cluster_mode yarn-client
```

Then open `https://localhost:6006` to see the result figures.

See [here](#Options) for more configurable options for this example.

# Options

* ```
  --cores The number of cores you want to use on each node. Default is 4.
  ```

* ```
  --num_nodes The number of nodes to be used in the cluster. Default is 1.
  ```

* ```
  --workers_per_node The number of workers to run on each node. Default is 2.
  ```

* ```
  --memory The memory you want to use on each node. Default is 10g.
  ```

* ```
  --epochs The number of epochs to train. Default is 2.
  ```

* ```
  --batch_size The worker batch size for training per executor. Default is 16.
  ```

# Results

You can find the results of training and validation

```
 Train stats: [{'num_samples': 60000, 'epoch': 1, 'batch_count': 3750, 'train_loss': 039022507713953654, 'last_train_loss': 0.2104504108428955}, {'num_samples': 60000, 'epoch': 2, 'batch_count': 3750, 'train_loss': 0.4949339405824741, 'last_train_loss': 0.18759065866470337}]
 
 Validation stats: {'num_samples': 10000, 'batch_count': 313, 'val_loss': 0.48876683268547055, 'last_val_loss': 0.4236503839492798, 'val_accuracy': 0.8217, 'last_val_accuracy': 0.75}
```


