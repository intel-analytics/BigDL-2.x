## Image Classification Python example
This example illustrates how to load and predict for Imagenet models with given image path
### Run steps
1. Build 

In order to run this example locally, we should build the project with `all-in-one` profile

mvn clean package -P all-in-one

2. Setup environment

Add model jar to `BIGDL_CLASSPATH` and python zip file to `PYTHONPATH`

export DL_PYTHON_HOME=${path_to_analytics-zoo}/models/target/models-0.1-SNAPSHOT-python-api.zip

export PYTHONPATH=$PYTHONPATH:${path_to_bigdl}/dist/conf/spark-bigdl.conf:$DL_PYTHON_HOME

export BIGDL_CLASSPATH=$(find ${path_to_bigdl}/dist/lib -name "*with-dependencies.jar" | head -n 1)
export BIGDL_CLASSPATH=$BIGDL_CLASSPATH:${path_to_analytics-zoo}/models/target/bigdl-models-0.1-SNAPSHOT-python-api.zip

3. Run the example

python ImageClassificationPredict.py [path_to_bigdl_model] [path_to_images] [top N number]
