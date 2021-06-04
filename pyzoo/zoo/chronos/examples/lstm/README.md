# AutoLSTM examples on mpg dataset
This example will demonstrate the effect of AutoLSTM on the mpg dataset on Seaborn. AutoLstm will return the best set of hyperparameters within the specified hyperparameter range.

## Prepare the environment
We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:
```
conda create -n zoo python=3.7 # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo
pip install torch==1.8.1 ray[tune]==1.2.0 seaborn scikit-learn
```

## Prepare data
mpg is the data describing the fuel economy of automobiles in the late 1970s and early 1980s. Here, only a few of these characteristics are selected for prediction.

## Run on local after pip install
```
python test_auto_lstm.py
```

## Run on yarn cluster for yarn-client mode after pip install 
```
python test_auto_lstm.py --cluster_model yarn
```

## Options
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`. You can refer to OrcaContext documents [here](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for details.
* `--num_workers` The number of workers to be used in the cluster. You can change it depending on your own cluster setting. Default to be 2.
* `--cores` "The number of cpu cores you want to use on each node. Default to be 4.
* `--memory` The memory you want to use on each node. Default to be 10g.
* `--backend` The backend of the lstm model. We only support backend as 'torch' for now.
* `--epoch` Max number of epochs to train in each trial. Default to be 1.
* `--cpus_per_train` Number of cpus for each trial. Default to be 2.
* `--n_sampling` Number of times to sample from the search_space. Default to be 1.