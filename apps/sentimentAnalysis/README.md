# Sentiment Analysis
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
* Python 2.7/3.5/3.6 (pandas 0.22.0)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo 0.1.0

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 12g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 12g

