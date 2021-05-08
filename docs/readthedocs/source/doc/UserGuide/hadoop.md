# Hadoop/YARN User Guide

Hadoop version: Hadoop >= 2.7 or CDH 5.X, Hadoop 3.X or CHD 6.X are not supported

---

You can run Analytics Zoo programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install Analytics Zoo or any Python libraries in the cluster).

### **1. Prepare Environment**

- You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment _**on the local client machine**_. Create a conda environment and install all the needed Python libraries in the created conda environment:

  ```bash
  conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
  conda activate zoo

  # Use conda or pip to install all the needed Python dependencies in the created conda environment.
  ```

- You need to download and install JDK in the environment, and properly set the environment variable `JAVA_HOME`, which is required by Spark. __JDK8__ is highly recommended.

  You may take the following commands as a reference for installing [OpenJDK](https://openjdk.java.net/install/):

  ```bash
  # For Ubuntu
  sudo apt-get install openjdk-8-jre
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

  # For CentOS
  su -c "yum install java-1.8.0-openjdk"
  export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.282.b08-1.el7_9.x86_64/jre

  export PATH=$PATH:$JAVA_HOME/bin
  java -version  # Verify the version of JDK.
  ```

- Check the Hadoop setup and configurations of your cluster. Make sure you properly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:

  ```bash
  export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
  ```

### **1.1 Setup for CDH**

CDH Version: Except 5.15.2, other CDH 5.X, CDH 6.X is not supported

---
### **2. YARN Client Mode on CDH**

Follow the steps below if you need to run Analytics Zoo in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

- Download and extract [Spark](https://spark.apache.org/downloads.html). You are recommended to use [Spark 2.4.3](https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz). Set the environment variable `SPARK_HOME`:

  ```bash
  export SPARK_HOME=the root directory where you extract the downloaded Spark package
  ```

- Download and extract [Analytics Zoo](../release.md). Make sure the Analytics Zoo package you download is built with the compatible version with your Spark. Set the environment variable `ANALYTICS_ZOO_HOME`:

  ```bash
  export ANALYTICS_ZOO_HOME=the root directory where you extract the downloaded Analytics Zoo package
  ```
- Before running the following two examples, please download [dataset of MNIST](http://yann.lecun.com/exdb/mnist/) on Cloudra Manager and CDH

#### (1) Python Example  
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  PYSPARK_PYTHON=./environment/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --master yarn \
    --deploy-mode client \
    --executor-cores 44 \
    --num-executors 3 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
  analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
    -f hdfs://172.16.0.105:8020/mnist \
    -b 132 \
    -e 3
  ```
If run success, you would see the output like:
> INFO  DistriOptimizer$:427 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Trained 132.0 records in 0.051549321 seconds. Throughput is 2560.6545 records/second. Loss is 0.15130447. Sequential4f65e3db's hyper parameters: Current learning rate is 0.05. Current dampening is 1.7976931348623157E308. <br>
> INFO  DistriOptimizer$:472 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Epoch finished. Wall clock time is 91485.329183 ms <br>
> INFO  DistriOptimizer$:111 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Validate model... <br>
> INFO  DistriOptimizer$:177 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] validate model throughput is 49120.7 records/second <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Top1Accuracy is Accuracy(correct: 9665, count: 10000, accuracy: 0.9665) <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Top5Accuracy is Accuracy(correct: 9995, count: 10000, accuracy: 0.9995) <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 91.114175155s] Loss is (Loss: 1116.1461, count: 10000, Average Loss: 0.111614615) <br>


#### (2) Scala Example
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  # Spark yarn client mode, please make sure the right HADOOP_CONF_DIR is set
  ${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh \
  --master yarn \
  --deploy-mode client \
  --executor-cores 44 \
  --num-executors 3 \
  --class com.intel.analytics.bigdl.models.lenet.Train \
  analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
  -f hdfs://172.16.0.105:8020/mnist \
  -b 132 \
  -e 3
  ```
If run success, you would see the output like:
> INFO  DistriOptimizer$:427 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Trained 132.0 records in 0.048059022 seconds. Throughput is 2746.6228 records/second. Loss is 0.10078872. Sequential20dc409's hyper parameters: Current learning rate is 0.05. Current dampening is 1.7976931348623157E308. <br>
> INFO  DistriOptimizer$:472 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Epoch finished. Wall clock time is 89554.313084 ms <br>
> INFO  DistriOptimizer$:111 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Validate model... <br>
> INFO  DistriOptimizer$:177 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] validate model throughput is 52652.59 records/second <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Top1Accuracy is Accuracy(correct: 9614, count: 10000, accuracy: 0.9614) <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Top5Accuracy is Accuracy(correct: 9995, count: 10000, accuracy: 0.9995) <br>
> INFO  DistriOptimizer$:180 - [Epoch 3 60060/60000][Iteration 1365][Wall Clock 89.182042038s] Loss is (Loss: 1263.0082, count: 10000, Average Loss: 0.12630081) <br>

---
### **5. YARN Cluster Mode on CDH**

Follow the steps below if you need to run Analytics Zoo in [YARN cluster mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

- Download and extract [Spark](https://spark.apache.org/downloads.html). You are recommended to use [Spark 2.4.3](https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz). Set the environment variable `SPARK_HOME`:

  ```bash
  export SPARK_HOME=the root directory where you extract the downloaded Spark package
  ```

- Download and extract [Analytics Zoo](../release.md). Make sure the Analytics Zoo package you download is built with the compatible version with your Spark. Set the environment variable `ANALYTICS_ZOO_HOME`:

  ```bash
  export ANALYTICS_ZOO_HOME=the root directory where you extract the downloaded Analytics Zoo package
  ```
- Before running the following two examples, please download [dataset of MNIST](http://yann.lecun.com/exdb/mnist/) on Cloudra Manager and CDH

#### (1) Python Example  
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  PYSPARK_PYTHON=./environment/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
    --master yarn \
    --deploy-mode cluster \
    --executor-cores 44 \
    --num-executors 3 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
  analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
    -f hdfs://172.16.0.105:8020/mnist \
    -b 132 \
    -e 3
  ```
If run success, you would see the output like:
> final status: SUCCEEDED

and then check the log detail using the following given URL in the output.

#### (2) Scala Example
- Use `spark-submit` to submit training LeNet example on CDH with Analytics Zoo:

  ```bash
  # Spark yarn cluster mode, please make sure the right HADOOP_CONF_DIR is set
  ${ANALYTICS_ZOO_HOME}/bin/spark-submit-scala-with-zoo.sh \
    --master yarn \
    --deploy-mode cluster \
    --executor-cores 44 \
    --num-executors 3 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    analytics-zoo/dist/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-jar-with-dependencies.jar \
    -f hdfs://172.16.0.105:8020/mnist \
    -b 132 \
    -e 3
  ```
If run success, you would see the output like:
> final status: SUCCEEDED

and then check the log detail using the following given URL in the output.
