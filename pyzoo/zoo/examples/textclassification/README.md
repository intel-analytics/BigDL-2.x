## Summary
This text classification example uses pre-trained GloVe embeddings to convert words to vectors,
and trains a CNN, LSTM or GRU `TextClassifier` model on 20 Newsgroup dataset.
It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

A CNN `TextClassifier` model can achieve around 85% accuracy after 20 epochs of training.
LSTM and GRU models are a little bit difficult to train, and more epochs are needed to achieve compatible results.

__Remark__: Due to some permission issue, this example cannot be run on Windows platform.


## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.


## Data Preparation
The data used in this example are:
- [20 Newsgroup dataset](http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz) which contains 20 categories and with 19997 texts in total.
- [GloVe word embeddings](http://nlp.stanford.edu/data/glove.6B.zip): embeddings of 400k words pre-trained on a 2014 dump of English Wikipedia.

Executing this example will automatically download and extract the data for you during the first run if no data has been detected in the target path.

You can also choose to prepare the data by yourself beforehand. The following scripts we prepare will serve to download and extract the data:
```bash
bash ${ANALYTICS_ZOO_HOME}/bin/data/news20/get_news20.sh dir
bash ${ANALYTICS_ZOO_HOME}/bin/data/glove/get_glove.sh dir
```
where `ANALYTICS_ZOO_HOME` is the folder where you extract the downloaded package and `dir` is the directory you wish to locate the downloaded data. If `dir` is not specified, the data will be downloaded to the current working directory. 20 Newsgroup dataset and GloVe word embeddings are supposed to be placed under the same directory.

The data folder structure after extraction should look like the following:
```
/tmp/text_data$ tree .
    .
    ├── 20news-18828
    └── glove.6B
```


## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=2g
python text_classification.py --data_path /tmp/data
```
See [here](#options) for more configurable options for this example.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.


## Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...

${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    text_classification.py --data_path /tmp/text_data
```
See [here](#options) for more configurable options for this example.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.


## Options
* `--data_path` The path where the training and word2Vec data locate. Default is `/tmp/text_data`. Make sure that you have write permission to the specified path if you want the program to automatically download the data for you.
* `--partition_num` The number of partitions to cut the dataset into. Default is 4.
* `--token_length` The size of each word vector. GloVe supports token_length 50, 100, 200 and 300. Default is 200.
* `--sequence_length` The length of a sequence. Default is 500.
* `--max_words_num` The maximum number of words. Default is 5000.
* `--encoder` The encoder for the input sequence. String, 'cnn' or 'lstm' or 'gru'. Default is 'cnn'.
* `--encoder_output_dim` The output dimension of the encoder. Default is 256.
* `--training_split` The split portion of the data for training. Default is 0.8.
* `-b` `--batch_size` The number of samples per gradient update. Default is 128.
* `--nb_epoch` The number of iterations to train the model. Default is 20.
* `-l` `--learning_rate` The learning rate for the TextClassifier model. Default is 0.01.
* `--log_dir` The path to store training and validation summary. Default is `/tmp/.bigdl`.
* `--model` Specify this option only if you want to load an existing TextClassifier model and in this case its path should be provided.


## Results
You can find the accuracy information from the log during the training process:
```
INFO  DistriOptimizer$:702 - [Epoch 20 15232/15133][Iteration 2720][Wall Clock 667.510906939s] Validate model...
INFO  DistriOptimizer$:744 - [Epoch 20 15232/15133][Iteration 2720][Wall Clock 667.510906939s] Top1Accuracy is Accuracy(correct: 3205, count: 3695, accuracy: 0.8673883626522327)
```
