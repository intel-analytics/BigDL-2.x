# Preprocess the Criteo dataset for DLRM Model
This example demonstrates how to use Analytics Zoo Friesian to preprocess the 
[Criteo](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) dataset to be used for [DLRM](https://arxiv.org/abs/1906.00091) model training.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo
```

## Prepare the data
You can download the full 1TB Click Logs dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).
 
After you download the files, you should convert them to parquet files with the name day_x.parquet(x=0-23), and put all parquet files in one folder.



## Running command
* Spark local, we can use the first several days to have a trial, example command
```bash
python dlrm_preprocessing.py \
    --ores 36 \
    --memory 50g \
    --days 0-1 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

* Spark standalone, example command
```bash
python dlrm_preprocessing.py \
    --cluster_mode standalone \
    --master spark://master/url \
    --cores 56 \
    --memory 240g \
    --num_nodes 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

* Spark yarn client mode, example command
```bash
python dlrm_preprocessing.py \
    --cluster_mode yarn \
    --cores 56 \
    --memory 240g \
    --num_nodes 8 \
    --days 0-23 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --frequency_limit 15
```

In the above commands
* cluster_mode: The cluster mode, such as local, yarn, or standalone. Default: local.
* master: The master URL, only used when cluster mode is standalone.
* cores: The executor core number. Default: 48.
* memory: The executor memory. Default: 160g.
* num_nodes: The number of executors. Default: 8.
* driver_cores: The driver core number. Default: 4.
* driver_memory: The driver memory. Default: 36g.
* days: Day range for preprocessing, such as 0-23, 0-1.
* input_folder: Path to the folder of parquet files.
* frequency_limit: Categories with a count/frequency below frequency_limit will be omitted from
 the encoding. For instance, '15', '_c14:15,_c15:16', etc. We recommend using "15" when you
  preprocess the whole 1TB dataset. Default: 15.
