# Java Spark Occlum

## Spark 2.4.3 local test
First, go to the spark_local directory and download Spark 2.4.3:
```
cd spark_local
wget https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
tar -xvzf spark-2.4.3-bin-hadoop2.7.tgz
cp spark-2.4.3-bin-hadoop2.7/jars/spark-network-common_2.11-2.4.3.jar spark-network-common_2.11-2.4.3.jar
```

[Download a BigDL release for spark 2.4.3](https://bigdl-project.github.io/master/#release-download/) and put `bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar` into the `spark_local` directory. 

[Download an Analytics Zoo release for spark 2.4.3](https://analytics-zoo.github.io/master/#release-download/) and put `analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar` into the `spark_local` directory. 

In `run_spark_on_occlum_glibc.sh`, configure `BIGDL_VERSION`, `SPARK_VERSION`, and `ANALYTICS_ZOO_VERSION`. In `start-occlum-spark.sh`, configure `LOCAL_IP`. 

In `init.sh`, configure your http and https proxies.

Pull the docker image:
```
docker pull occlum/occlum:0.23.0-ubuntu18.04
```

To train a model with PPML in Analytics Zoo and BigDL, you need to prepare the data first. The Docker image is taking lenet and MNIST as example.

You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in a new directory `data`. There are four files in total. `train-images-idx3-ubyte` contains train images, `train-labels-idx1-ubyte` is train label file, `t10k-images-idx3-ubyte` has validation images and `t10k-labels-idx1-ubyte` contains validation labels. 

To run Spark pi example, start the docker container with:
```
bash start-occlum-spark.sh test
```
To run BigDL example, start the docker container with:
```
bash start-occlum-spark.sh bigdl
```
The examples are run in the docker container. Attach it and see the results.

