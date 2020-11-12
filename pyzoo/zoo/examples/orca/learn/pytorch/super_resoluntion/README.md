# Orca PyTorch estimator on BSDS300 dataset

This is an example to show you how to use analytics-zoo orca PyTorch Estimator to implement super-resolution.

# Requirements
* Python 3.7
* torch 1.5.0 or above
* torchvision 0.6.0 or above
* Apache Spark 2.4.3(pyspark)
* jep 3.9.0
* PIL 8.0.0

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install jep==3.9.0
pip install pillow
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
conda activate zoo
python super_resolution.py --upscale_factor 3 --mode local
```

- Run with Yarn Client mode:
```bash
conda activate zoo
python super_resolution.py --upscale_factor 3 --mode yarn --node 4 --dataset ./dataset/dataset
```

In above commands
* --upscale_factor: super resolution upscale factor.
* --batchSize: training batch size.
* --testBatchSize: testing batch size.
* --lr: learning Rate.
* --nEpochs: number of epochs to train for.
* --mode: the mode of spark cluster. support local and yarn.
* --dataset: the dir of dataset.
* --node: number of working nodes.
