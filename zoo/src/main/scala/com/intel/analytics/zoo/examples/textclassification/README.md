## Summary
This text classification example uses pre-trained GloVe embeddings to convert words to vectors,
and trains a CNN, LSTM or GRU `TextClassifier` model on 20 Newsgroup dataset.
It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

A CNN `TextClassifier` model can achieve around 85% accuracy after 20 epochs of training.
LSTM and GRU models are a little bit difficult to train, and more epochs are needed to achieve compatible results.


## Data Preparation
The data used in this example are:
- [20 Newsgroup dataset](http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz) which contains 20 categories and with 18828 texts in total.
- [GloVe word embeddings](http://nlp.stanford.edu/data/glove.6B.zip): embeddings of 400k words pre-trained on a 2014 dump of English Wikipedia.

You need to prepare the data by yourself beforehand. The following scripts we prepare will serve to download and extract the data for you:
```bash
bash ${ANALYTICS_ZOO_HOME}/bin/data/news20/get_news20.sh dir
bash ${ANALYTICS_ZOO_HOME}/bin/data/glove/get_glove.sh dir
```
where `ANALYTICS_ZOO_HOME` is the `dist` directory under the Analytics Zoo project and `dir` is the directory you wish to locate the downloaded data. If `dir` is not specified, the data will be downloaded to the current working directory. 20 Newsgroup dataset and GloVe word embeddings are supposed to be placed under the same directory `baseDir`.

The data folder structure after extraction should look like the following:
```
baseDir$ tree .
    .
    ├── 20news-18828
    └── glove.6B
```


## Run this example
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
SPARK_HOME=the root directory of Spark
ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project
MASTER=...
ANALYTICS_ZOO_JAR=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_BIGDL_VERSION-spark_SPARK_VERSION-ZOO_VERSION-jar-with-dependencies.jar
BASE_DIR=the base directory containing the News20 and GloVe data

spark-submit \
    --master ${MASTER} \
    --driver-memory 2g \
    --executor-memory 2g \
    --class com.intel.analytics.zoo.examples.textclassification.TextClassification \
    ${ANALYTICS_ZOO_JAR} \
    --baseDir ${BASE_DIR}
```
__Options:__
* `--baseDir` This option is __required__. The path where the News20 and GloVe data locate.
* `--partitionNum` The number of partitions to cut the dataset into. Datault is 4.
* `--tokenLength` The size of each word vector. GloVe supports tokenLength 50, 100, 200 and 300. Default is 200.
* `--sequenceLength` The length of a sequence. Default is 500.
* `--maxWordsNum` The maximum number of words. Default is 5000.
* `--encoder` The encoder for the input sequence. String, 'cnn' or 'lstm' or 'gru'. Default is 'cnn'.
* `--encoderOutputDim` The output dimension of the encoder. Default is 256.
* `--trainingSplit` The split portion of the data for training. Default is 0.8.
* `-b` `--batchSize` The number of samples per gradient update. Default is 128.
* `--nbEpoch` The number of iterations to train the model. Default is 20.
* `-l` `--learningRate` The learning rate for the TextClassifier model. Default is 0.01.
* `--model` Specify this option only if you want to load an existing TextClassifier model and in this case its path should be provided.


## Results
You can find the accuracy information from the log during the training process:
```
INFO DistriOptimizer$: [Epoch 20 15120/15044][Iteration 2700][Wall Clock 643.613424167s] Validate model...
INFO DistriOptimizer$: [Epoch 20 15120/15044][Iteration 2700][Wall Clock 643.613424167s] Top1Accuracy is Accuracy(correct: 3207, count: 3784, accuracy: 0.8475158562367865)
```
