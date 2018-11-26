## Summary
This QA Ranker example uses pre-trained GloVe embeddings to convert words to vectors,
and trains a [KNRM](https://arxiv.org/abs/1706.06613) model on WikiQA dataset.


## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.


## Data Preparation
The data used in this example are:
- [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419): a new publicly available set of question and sentence pairs.
- Word Embeddings: We use [`glove.840B.300d.txt`](http://nlp.stanford.edu/data/glove.840B.300d.zip) in this example. You can also choose other available GloVe embeddings from [here](https://nlp.stanford.edu/projects/glove/).

For WikiQA dataset, we refer to MatchZoo to process it into corpus and relations: `question_corpus.csv`, `answer_corpus.csv`, `relation_train.csv` and `relation_valid.csv`.


## Run this example
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
dataDir=the directory containing corpus and relations of WikiQA
glovePath=the file path to GloVe embeddings

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 3g \
    --executor-memory 3g \
    --class com.intel.analytics.zoo.examples.qaranker.QARanker \
    --dataPath ${dataDir} \
    --embeddingFile ${glovePath}
```
See [here](#options) for more configurable options for this example.


## Options
* `--dataPath` This option is __required__. The directory containing the corpus and relations.
* `--embeddingFile` This option is __required__. The file path to GloVe embeddings.
* `--text1Length` The length of each question. Default is 10.
* `--text2Length` The length of each answer. Default is 40.
* `--tokenLength` The size of each word vector. GloVe supports tokenLength 50, 100, 200 and 300. Default is 300 for `glove.840B.300d.txt`.
* `-b` `--batchSize` The number of samples per gradient update. Default is 200.
* `--nbEpoch` The number of iterations to train the model. Default is 30.
* `-l` `--learningRate` The learning rate for the KNRM model. Default is 0.001.
* `--partitionNum` The number of partitions to cut the dataset into. Datault is 4.
* `--model` Specify this option only if you want to load an existing KNRM model and in this case its path should be provided.


## Results
You can find the validation information from the log during the training process:
```
INFO  TextMatcher$:86 - ndcg@3: 0.6417297245909217
INFO  TextMatcher$:86 - ndcg@5: 0.688879313318335
INFO  TextMatcher$:77 - map: 0.6373270433829106
```
