# Demo Setup Guide
This notebook demostrates how to build neural network recommendation system with explict feedback using BigDL on Spark. 

## Environment
* Python 2.7
* JDK 8
* Scala 2.11
* Apache Spark 2.0 above
* Jupyter Notebook 4.1
* BigDL 0.3.0
* [Setup env on Mac OS](https://github.com/intel-analytics/BigDL-Tutorials/blob/master/SetupMac.md) / [Setup env on Linux](https://github.com/intel-analytics/BigDL-Tutorials/blob/master/SetupLinux.md)

## Start Toree Notebook
* TODO, update the link
* Download [BigDL 0.3.0 on spark 2.0 or 2.1](https://bigdl-project.github.io/master/#release-download/) and unzip file.
* Run export BIGDL_HOME=where is your unzipped bigdl folder
* Run export SPARK_HOME=where is your unpacked spark folder
* Create start_notebook.sh, copy and paste the contents below, and edit SPARK_HOME, BigDL_HOME accordingly. Change other parameter settings as you need. 

```bash
#!/bin/bash

export SPARK_HOME=...
BigDL_HOME=...
BigDL_JAR_PATH=${BigDL_HOME}/lib/bigdl*.jar

jupyter toree install --spark_home=${SPARK_HOME}
SPARK_OPTS="--master=local[8] --driver-memory 10g  --executor-memory 20g  --jars ${BigDL_JAR_PATH}"  TOREE_OPTS="--nosparkcontext" jupyter notebook

```
* Put start_notebook.sh in home directory and execute them in bash.


