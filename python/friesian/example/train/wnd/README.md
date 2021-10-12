# Train the Twitter Recsys Challenge 2021 dataset for WideAndDeep Model
This example demonstrates how to use BigDL Friesian to train [WideAndDeep](https://arxiv.org/abs/1606.07792) model with the
[Twitter Recsys Challenge 2021](https://recsys-twitter.com/data/show-downloads#) dataset.

## Prepare the environment
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster (yarn-client mode only).
```
conda create -n zoo python=3.7  # "zoo" is the conda environment name, you can use any name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo
```

## Prepare the data
You can download the full Twitter dataset from [here](https://recsys-twitter.com/data/show-downloads#), which includes close to 1 billion data points, >40 million each day over 28 days.
 Week 1 - 3 will be used for training and week 4 for evaluation and testing. Each datapoint contains the tweet along with engagement features, user features, and tweet features.

After you download and decompress the files, there is a train parquet directory and validation csv file.
* For train data, execute this command to do some conversion:
```bash
python convert_train.py \
    --input_folder /path/to/the/folder/of/train_parquet_folder \
    --output_foler /path/to/the/folder/to/save/preprocessed
```
__Options:__
* `input_folder`: The path to the folder of train parquet folder.
* `output_folder`: The path to the folder of save the preprocessed data.

* For validation data, execute this command to change csv to parquet files:
```bash
python valid_to_parquet.py \
    --input_folder /path/to/the/folder/of/valid_csv \
    --output_foler /path/to/the/folder/to/save/preprocessed
```
__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 44.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 30.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `input_folder`: The path to the folder of validation csv file.
* `output_folder`: The path to the folder of save the preprocessed data.

## Running command

### Preprocess train and test data for WND training.

In converted train parquet directory, there are the first 21 days (day0 to day20) data used for WND training in total.
The files are with the name "part-00xxx.parquet" (x=000-269).

* Spark local, we can use the first several partitions train data to have a trial, example command:
```bash
python wnd_preprocessing_recsys.py \
    --executor_cores 36 \
    --executor_memory 50g \
    --train_files 0-1 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_foler /path/to/the/folder/to/save/preprocessed/parquet_files \
    --cross_sizes 600
```

* Spark standalone, example command to run on the full dataset:
```bash
python wnd_preprocessing_recsys.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 44 \
    --executor_memory 150g \
    --num_executor 8 \
    --train_files 0-269 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_foler /path/to/the/folder/to/save/preprocessed/parquet_files \
    --cross_sizes 600
```

* Spark yarn client mode, example command to run on the full dataset:
```bash
python wnd_preprocessing_recsys.py \
    --cluster_mode yarn \
    --executor_cores 44 \
    --executor_memory 150g \
    --num_nodes 8 \
    --train_files 0-269 \
    --input_folder /path/to/the/folder/of/parquet_files \
    --output_foler /path/to/the/folder/to/save/preprocessed/parquet_files \
    --cross_sizes 600
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 44.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 130g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `train_files`: The part range for train data preprocessing, such as 0-269 for the full training dataset,
 0-1 for the first two parts, etc. Default to be 0-269.
* `test_files`: The part range for test data preprocessing, such as 0-63 for the full testing dataset,
 0-1 for the first two parts, etc. Default to be 0-63.
* `input_folder`: The path to the folder of parquet files, either a local path or an HDFS path.
* `output_folder`: The path to save the preprocessed data to parquet files. HDFS path is recommended.
* `cross_sizes`: bucket size for cross column (`present_media_language`). Default to be 600.


### WND training from preprocessed train and test data.
* Spark local, we can use preprocessed small data to have a trial, example command:
```bash
python wnd_train_recsys.py \
    --executor_cores 36 \
    --executor_memory 30g \
    --data_dir /path/to/the/folder/of/preprocessed_parquet_files \
    --model_dir /path/to/the/folder/to/save/trained_model \
    --batch_size 3200 \
    --epoch 10 \
    --learning_rate 1e-4 \
    --early_stopping 3
```

* Spark standalone, example command to train on the full dataset:
```bash
python wnd_train_recsys.py \
    --cluster_mode standalone \
    --master spark://master-url:port \
    --executor_cores 44 \
    --executor_memory 30g \
    --num_executor 8 \
    --data_dir /path/to/the/folder/of/preprocessed_parquet_files \
    --model_dir /path/to/the/folder/to/save/trained_model \
    --batch_size 102400 \
    --epoch 2 \
    --learning_rate 1e-4
```

* Spark yarn client mode, example command to train on the full dataset:
```bash
python wnd_train_recsys.py \
    --cluster_mode yarn \
    --executor_cores 44 \
    --executor_memory 30g \
    --num_nodes 8 \
    --data_dir /path/to/the/folder/of/preprocessed_parquet_files \
    --model_dir /path/to/the/folder/to/save/trained_model \
    --batch_size 102400 \
    --epoch 2 \
    --learning_rate 1e-4
```

__Options:__
* `cluster_mode`: The cluster mode to run the data preprocessing, one of local, yarn or standalone. Default to be local.
* `master`: The master URL, only used when cluster_mode is standalone.
* `executor_cores`: The number of cores to use on each node. Default to be 48.
* `executor_memory`: The amount of memory to allocate on each node. Default to be 240g.
* `num_nodes`: The number of nodes to use in the cluster. Default to be 8.
* `driver_cores`: The number of cores to use for the driver. Default to be 4.
* `driver_memory`: The amount of memory to allocate for the driver. Default to be 36g.
* `data_dir`: The path to save the preprocessed data to parquet files. Either a local path or an HDFS path.
* `model_dir`: The path to the folder to save trained model.
* `batch_size`: The batch size for training. Default to 102400.
* `epoch`: Number of epochs to train. Default to 2.
* `learning_rate`: The initial learning rate for training. Default to 1e-4.
* `early_stopping`: The patience epochs for stopping training. Default to 3.
* `hidden_units`: Hidden units for deep mlp. Default to 1024, 1024
