# Summary
This example uses pre-trained GloVe embeddings to convert words to vectors,
and trains a [KNRM](https://arxiv.org/abs/1706.06613) model to solve question answering task
on WikiQA dataset.


## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.


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


## Run after pip install
You can easily use the following commands to run this example:
```bash
export SPARK_DRIVER_MEMORY=3g
data_dir=the directory that contains corpus and relations csv files listed above
glove_path=the file path to GloVe embeddings

python qa_ranker.py --data_path ${data_dir} --embedding_file ${glove_path}
```
See [here](#options) for more configurable options for this example.

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.


## Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
MASTER=...
data_dir=the directory that contains corpus and relations csv files listed above
glove_path=the file path to GloVe embeddings

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 3g \
    --executor-memory 3g \
    qa_ranker.py --data_path ${data_dir} --embedding_file ${glove_path}
```
See [here](#options) for more configurable options for this example.


## Options
* `--data_path` This option is __required__. The directory containing the corpus and relations.
* `--embedding_file` This option is __required__. The file path to GloVe embeddings.
* `--output_path` If specified, the trained model `knrm.model` and word dictionary file `word_index.txt` will be saved under this path. It can be either a local or distributed file system path.
* `--question_length` The sequence length of each question. Default is 10.
* `--answer_length` The sequence length of each answer. Default is 40.
* `--partition_num` The number of partitions to cut the datasets into. Default is 4.
* `-b` `--batch_size` The number of samples per gradient update. Default is 200.
* `-e` `--nb_epoch` The number of iterations to train the model. Default is 30.
* `-l` `--learning_rate` The learning rate for the model. Default is 0.001.
* `-m` `--model` Specify this option only if you want to load an existing KNRM model and in this case its path should be provided.


## Results
We use [NDCG](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Discounted_cumulative_gain) and [MAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) to evaluate the performance. These metrics are commonly used for ranking tasks.

You can find the validation information from the console log during the training process:
```
INFO  Ranker$:101 - ndcg@3: 0.627045395285388
INFO  Ranker$:101 - ndcg@5: 0.6863500126591011
INFO  Ranker$:83 - map: 0.6385866141096943
```
