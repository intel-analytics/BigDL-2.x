# PPML (Privacy Preserving Machine Learning)

PPML (Privacy-Preserving Machine Learning) aims at protecting user privacy, meanwhile keep machine learning applications still useful. However, achieving this goal without impacting existing applications is no trival, especially in end-to-end big data scenarios. To reslove this problem, Analytics-Zoo provides an end-to-end PPML platform for Big Data AI based on Intel SGX. This PPML platform ensure the whole Big Data & AI pipeline are fully protected by secured SGX enclave in hardware level, meanwhile exising Big Data & AI applications, such as Flink, Spark, SparkSQL and machine/deep learning, can be seamlessly migrated into this PPML platform without any code changes.



## PPML for Big Data AI

Based on Intel SGX (Software Guard Extensions) and LibOS projects ([Graphene](https://grapheneproject.io/) and [Occlum](https://occlum.io/)), Analytics-Zoo empowers our customers (e.g., data scientists and big data developers) to build PPML applications on top of large scale dataset without impacting existing applications.

One of the biggest challenge in PPML is migrating E2E

To take full advantage of big data, especially the value of private or sensitive data, customers need to build a trusted platform under the guidance of privacy laws or regulation, such as [GDPR](https://gdpr-info.eu/). This requirement raises a big challenge to customers who already have big data and big data applications, such as Spark/SparkSQL, Flink and AI applications. Migrating these applications into privacy preserving way requires lots of additional efforts.




Note: Intel SGX requires hardware support, please [check if your CPU has this feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). In [3rd Gen Intel Xeon Scalable Processors](https://newsroom.intel.com/press-kits/3rd-gen-intel-xeon-scalable/), SGX allows up to 1TB of data to be included in secure enclaves.

### Key features

- Protecting data and model confidentiality
  - Sensitive input/output data (computation, training and inference), e.g., healthcare data
  - Propretary model, e.g., model trained with self-owned or sensitive data
- Seamless migrate existing big data applications into privacy preserving applications
- Trusted big data & AI Platform based on Intel SGX
  - Trusted Big Data Analytics and ML
  - Trusted Realtime Compute and ML

## Trusted Big Data Analytics and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Spark in SGX, then run Spark PI example in safe way. For more examples, please refer to [trusted-big-data-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene).

### Use cases

- Big Data analysis using Spark (Spark SQL, Dataframe, MLlib, etc.)
- Distributed deep learning using BigDL

### Get started


#### Prerequisite

Please check if current platform [has SGX feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). Then, enable SGX feature in BIOS. Note that after SGX is enabled, a protion of memory will be assigned to SGX, and cannot be used or seen by OS. 

Download scripts and dockerfiles in [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

Check SGX driver with `ls /dev | grep sgx`. If SGX driver is not installed, run `install-graphene-driver.sh` and run it with root premission.

```bash
./ppml/scripts/install-graphene-driver.sh
```

#### Setp 0: Prepare Environment

Prepare keys for TLS.

```bash
./ppml/scripts/generate-keys.sh
```

Build docker image from Dockerfile

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
cp -r ../../keys .
./build-docker-image.sh
```

#### Step 1: Start Spark in SGX

Enter `analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene` dir. Start Spark service with this command

```bash
./start-local-big-data-ml.sh
sudo docker exec -it spark-local bash
```

##### Step 2: Submit jobs to Spark

Modify `init.sh`, which is task/env related.

```bash
./init.sh
vim start-spark-local-pi-sgx.sh
```

Add these lines in the `start-spark-local-pi-sgx.sh` file:

```bash
#!/bin/bash

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
        -Xmx10g \
        -Dbigdl.mklNumThreads=1 \
        -XX:ActiveProcessorCount=24 \
        org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class org.apache.spark.examples.SparkPi \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar | tee spark.local.pi.sgx.log
```

Then run the script to run pi test in spark:

```bash
chmod a+x start-spark-local-pi-sgx.sh
./start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like:

>   Pi is roughly 3.1422957114785572

This example is a simple Spark local PI example, if you want to run application in distributed Spark cluster protected by SGX, please refer to [distributed bigdata ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene#start-the-distributed-bigdata-ml).

## Trusted Realtime Compute and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Flink in SGX, then run real-time model serving in safe way. For more examples, please refer to [trusted-realtime-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene).

### User cases

- Real time data computation/analytics using Flink
- Distributed end-to-end serving solution with Cluster Serving

### Get started

#### Setp 0: Prepare Environment

Please clone or download Analytics-Zoo source code, then enter `analytics-zoo/ppml`. If SGX driver is not installed, please install SGX driver with this command.

```bash
./scripts/install-graphene-driver.sh
```

Prepare the keys for TLS.

```bash
./scripts/generate-keys.sh
```

Build docker image from Dockerfile.

```bash
cd trusted-realtime-ml/scala/docker-graphene
cp -r ../../keys .
./build-docker-image.sh
```

#### Step 1: Start Cluster Serving Service

Enter `analytics-zoo/ppml/trusted-realtime-ml/scala/docker-graphene` dir.

Start cluster serving in single container

```bash
./start-local-cluster-serving.sh
```

#### Step 2: Inference with Cluster Serving

After all services are ready, then you can directly push inference requests int queue with [Restful API](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/#restful-api). Also, you can push image/input into queue with Python API

```python
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```

#### Step 3: Stop Cluster Serving Service

Cluster Serving service is a long running service in container, you can stop it with

```bash
docker stop containerID
```
