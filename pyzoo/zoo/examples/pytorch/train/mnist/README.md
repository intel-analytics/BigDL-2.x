# Distributed pytorch on MNIST dataset

This is an example to show you how to use analytics-zoo to train a pytorch model on Spark. 

# Requirements
* Python 3.7
* torch 1.5.0
* torchvision 0.6.0
* Apache Spark 2.4.3(pyspark)

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the enviroments, especially if you want to run on a yarn cluster(yarn-client mode only). 
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
conda install pytorch-cpu torchvision-cpu -c pytorch
```

## Run example
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode
You can easily use the following commands to run this example:
    ```bash
    conda activate zoo
    export ZOO_NUM_MKLTHREADS=4
    python main.py
    ```

- Run with Yarn Client mode, upload data to hdfs first, export env `HADOOP_CONF_DIR` and `ZOO_CONDA_NAME`:  
    ```bash
    conda activate zoo
    hdfs dfs -put /tmp/zoo/dogs_cats dogs_cats 
    export HADOOP_CONF_DIR=[path to your hadoop conf directory who has yarn-site.xml]
    export ZOO_CONDA_NAME=zoo #conda environment name you just prepared above
    export ZOO_NUM_MKLTHREADS=all
    python main.py
    ```
    
In above commands
* --batch-size: The mini-batch size on each executor.
* --test-batch-size: The test's mini-batch size on each executor.
* --lr: learning rate.
* --epochs: number of epochs to train.
* --seed: random seed.
* --save-model: for saving the current model.
