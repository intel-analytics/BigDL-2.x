# Orca PyTorch Super Resolution example on BSDS300 dataset

We demonstrate how to easily run synchronous distributed Pytorch training using Pytorch Estimator of Project Orca in Analytics Zoo. This is an example using the efficient sub-pixel convolution layer to train on BSDS3000 dataset, using crops from the 200 training images, and evaluating on crops of the 100 test images. See [here](https://github.com/pytorch/examples/tree/master/super_resolution) for the original single-node version of this example provided by Pytorch.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo[ray]  # 0.9.0 or above
pip install pillow
conda install pytorch torchvision cpuonly -c pytorch  # command for linux
conda install pytorch torchvision -c pytorch  # command for macOS
```

## Prepare Dataset
You can specify runtime parameter `--download` in parser, then dataset will be auto-downloaded on each node.
If your yarn nodes can't access internet, run the `prepare_dataset.sh` to prepare dataset automatically.
```
bash prepare_dataset.sh
```
After the script runs, you will see folder **dataset (for local mode use)** and archive **dataset.zip (for yarn mode use)**.

## Run example
You can run this example on local mode and yarn-client mode.

- Run with Spark Local mode:
```bash
python super_resolution.py --cluster_mode local
```

- Run with Yarn-Client mode:
```bash
python super_resolution.py --cluster_mode yarn
```

In above commands
* `--upscale_factor` The upscale factor of super resolution. Default is 3.
* `--batch_size` The number of samples per gradient update. Default is 16.
* `--test_batch_size` The number of samples per batch validate. Default is 100.
* `--lr` Learning Rate. Default is 0.001.
* `--epochs` The number of epochs to train for. Default is 2.
* `--cluster_mode` The mode of spark cluster. support local and yarn. Default is "local".

## Results
You can find the logs for training:
```
train stats==> num_samples:200 ,epoch:10 ,batch_count:13 ,train_loss0.00432804074138403 ,last_train_loss0.0033895261585712433
```
And after validation, test results will be seen like:
```
validation stats==> num_samples:100 ,batch_count:1 ,val_loss0.10453934967517853 ,last_val_loss0.10453934967517853
```