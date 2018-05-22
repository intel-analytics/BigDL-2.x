# Demo Setup Guide
This is a sample of sentiment analysis using Analytics Zoo and BigDL on Spark. 
In this example, you learned how to use BigDL to develop deep learning models for sentiment analysis including:
* How to load and review the IMDB dataset
* How to do word embedding with Glove
* How to build a CNN model for NLP with BigDL
* How to build a LSTM model for NLP with BigDL
* How to build a GRU model for NLP with BigDL
* How to build a Bi-LSTM model for NLP with BigDL
* How to build a CNN-LSTM model for NLP with BigDL
* How to train deep learning models with BigDL

## Environment
* Python 2.7
* JDK 8
* Scala 2.11 
* Apache Spark 2.x
* Jupyter Notebook 4.1
* Zoo 0.1.0

## Steps to run the notebook
* Run `export ANALYTICS_ZOO_HOME=the home directory of the Analytics Zoo project`
* Run `export SPARK_HOME=the root directory of Spark`
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie MASTER = local\[physcial_core_number\]
```bash
MASTER=local[*]
bash ${ANALYTICS_ZOO_HOME}/scripts/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g \
```

