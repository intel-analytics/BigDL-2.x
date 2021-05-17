# PyTorch Sentiment example

We demonstrate how to easily show the results of running synchronous distributed PyTorch training using PyTorch Estimator of Project Orca in Analytics Zoo. We use the LSTMClassifier to train on IMDB dataset. See [here](https://github.com/prakashpandey9/Text-Classification-Pytorch) for the original single-node version of this example. We provide two distributed PyTorch training backends for this example, namely "bigdl" and "torch_distributed". You can run with either backend as you wish.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install torch==1.7.1
pip install torchtext==0.8.0

# For bigdl backend:
pip install analytics-zoo  # 0.10.0.dev3 or above
pip install jep==3.9.0
pip install six cloudpickle

# For torch_distributed backend:
pip install analytics-zoo[ray]  # 0.10.0.dev3 or above
```

## Run on local after pip install

The default backend is `torch_distributed`.

```
python main.py
```

You can also run with `bigdl` backend via:

```
python main.py --backend bigdl
```

## Run on yarn cluster for yarn-client mode after pip install

```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python main.py --cluster_mode yarn-client
```

The default backend is `torch_distributed`. You can also run with `bigdl` by specifying the backend.

## Results

**For "bigdl" backend**

You can find the logs for training as follows:

Final test results will be printed at the end:

**For "torch_distributed" backend**

Final test results will be printed at the end:

```
num_samples : 24960
Accuracy : tensor(0.8286)
val_loss : 0.387816492582743
```

