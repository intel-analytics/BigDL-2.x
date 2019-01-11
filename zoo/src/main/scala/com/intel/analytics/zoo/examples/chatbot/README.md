## Summary
This example demonstrates how to train a chatbot and use it to inference answers for queries.

## Preparation

To start with this example, you need prepare your dataset.

1. Prepare training dataset

    The dataset can be downloaded from https://s3.amazonaws.com/analytics-zoo-data/chatbot-data.tar.gz.
    
    ```bash
    wget https://s3.amazonaws.com/analytics-zoo-data/chatbot-data.tar.gz
    tar zxvf chatbot-data.tar.gz
    ```

## Run this example

Command to run the example in Spark local mode:
```bash
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
