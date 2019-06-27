# Face Generation using TorchNet

This example demonstrates how to run inference using a pre-trained Pytorch Model from Pytorch Hub.

## Environment
* Pytorch & TorchVision 1.1.0
* Apache Spark 1.6.x/2.x (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics-Zoo 0.6.0-SNAPSHOT and above
* Jupyter Notebook

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) 
to install analytics-zoo via __download the prebuilt package__.

**Notice**: This new feature is included in 0.6.0-SNAPSHOT, so before Analytics-Zoo release 0.6.0, pip install won't work.  


## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
bash ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 10g
```
