## Summary
This example demonstrates how to train a chatbot and use it to inference answers for queries.

## Download Analytics Zoo
You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.

## Data Preparation

To start with this example, you need prepare your dataset.

Prepare training dataset

    The dataset can be downloaded from https://s3.amazonaws.com/analytics-zoo-data/chatbot-data.tar.gz.
    
    ```bash
    wget https://s3.amazonaws.com/analytics-zoo-data/chatbot-data.tar.gz
    tar zxvf chatbot-data.tar.gz
    ```
After unzip the file, you will get 4 files.

idx2w.csv: dictionary which used to map words to indexes.

w2idx.csv: dictionary which used to index words

chat1_1.txt: Queries. Each line is a query. The file has been indexed with dictionray

chat2_1.txt: Answers to the queries. The file has been indexed with dictionray


## Run this example

Command to run the example in Spark local mode:
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package
dataPath=... // data path. Local file system is supported.

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--verbose \
--master $master \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.chatbot.Train \
-f ${dataPath}
```

### Results
You can find infered answers for given query at end of each epoch.

Query> happy birthday have a nice day

SENTENCESTART thank you  SENTENCEEND

Query> donald trump won last nights presidential debate according to snap online polls

SENTENCESTART i know that SENTENCEEND
