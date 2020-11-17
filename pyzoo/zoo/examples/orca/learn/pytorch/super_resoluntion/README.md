# Orca PyTorch estimator on BSDS300 dataset

This is an example to show you how to use analytics-zoo orca PyTorch Estimator to implement [super-resolution](https://github.com/leonardozcm/examples/tree/master/super_resolution).

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.3
* ray 0.8.4
* PIL 8.0.0

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install pillow
pip install ray==0.8.4
conda install pytorch torchvision cpuonly -c pytorch #command for linux
conda install pytorch torchvision -c pytorch #command for macOS
```

## Prepare Dataset
If your nodes can access internet, run the `prepare_dataset.sh` to prepare dataset automatically.
```
bash prepare_dataset.sh
```
After the script runs, you will see folder **dataset(for local mode use)** and archive **dataset.zip(for yarn mode use)**.

## Run example
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode:
```bash
python super_resolution.py --upscale_factor 3 --cluster_mode local
```

- Run with Yarn Client mode:
```bash
python super_resolution.py --upscale_factor 3 --cluster_mode yarn --num_nodes 4 --dataset ./dataset/dataset
```

In above commands
* --upscale_factor: super resolution upscale factor. Default is 3.
* --batchSize: training batch size. Default is 16.
* --testBatchSize: testing batch size. Default is 100.
* --lr: learning Rate. Default is 0.001.
* --epochs: number of epochs to train for. Default is 2.
* --cluster_mode: the mode of spark cluster. support local and yarn. Default is "local".
* --dataset: the dir of dataset. Default is "./dataset".
* --num_nodes: The number of nodes to be used in the cluster. Default is 1.
* --cores:The number of cpu cores you want to use on each node. Default is 4.
* --memory:The memory you want to use on each node. Default is 2g.
