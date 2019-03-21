# Summary
This example uses pre-trained GloVe embeddings to convert words to vectors,
and trains a [KNRM](https://arxiv.org/abs/1706.06613) model to solve question answering task 
on WikiQA dataset.


## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.


## Data Preparation
__QA Dataset:__
- For convenience, you are __recommended to directly download__ our processed WikiQA dataset from [here](https://s3.amazonaws.com/analytics-zoo-data/WikiQAProcessed.zip) and unzip it.
- [WikiQA](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a new publicly available set of question and sentence pairs.
- Instead of using original WikiQA dataset format directly, we refer to [MatchZoo](https://github.com/NTMC-Community/MatchZoo) to process raw data into corpus and relations.
Thus this example expects the following input files put under the same directory, which ought to applicable for general question answering tasks:
    - `question_corpus.csv`: Each record contains QuestionID and content separated by comma.
    - `answer_corpus.csv`: Each record contains AnswerID and content separated by comma.
    - `relation_train.csv` and `relation_valid.csv`: Question and answer correspondence for training and validation respectively. Each record contains QuestionID, AnswerID and label (0 or 1) separated by comma.
- If you wish, you can also follow similar steps listed in this [script](https://github.com/NTMC-Community/MatchZoo/blob/v1.0/data/WikiQA/run_data.sh) to process the raw WikiQA dataset.

__Word Embeddings:__
- We use [`glove.840B.300d.txt`](http://nlp.stanford.edu/data/glove.840B.300d.zip) in this example.
- You can also choose other available GloVe embeddings from [here](https://nlp.stanford.edu/projects/glove/).


## Run this example
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
dataDir=the directory that contains corpus and relations csv files listed above
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
* `--questionLength` The sequence length of each question. Default is 10.
* `--answerLength` The sequence length of each answer. Default is 40.
* `--partitionNum` The number of partitions to cut the datasets into. Default is 4.
* `-b` `--batchSize` The number of samples per gradient update. Default is 200.
* `-e` `--nbEpoch` The number of iterations to train the model. Default is 30.
* `-l` `--learningRate` The learning rate for the model. Default is 0.001.
* `--memoryType` Memory type used for caching training data. Default is `DRAM`. You can change it to `PMEM` if you have Intel Optane DC Persistent Memory.
* `-m` `--model` Specify this option only if you want to load an existing KNRM model and in this case its path should be provided.



## Results
We use [NDCG](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Discounted_cumulative_gain) and [MAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) to evaluate the performance. These metrics are commonly used for ranking tasks.

You can find the validation information from the console log during the training process:
```
INFO  Ranker$:103 - ndcg@3: 0.6449269813235653
INFO  Ranker$:103 - ndcg@5: 0.6953062444306408
INFO  Ranker$:84 - map: 0.6560713265739151
```
