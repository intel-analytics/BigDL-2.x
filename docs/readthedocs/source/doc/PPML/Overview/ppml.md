# PPML (Privacy Preserving Machine Learning)

PPML (Privacy-Preserving Machine Learning) aims at protecting user privacy, meanwhile keep machine learning applications still useful. However, achieving this goal without impacting existing applications is difficult, especially in end-to-end big data scenarios. To reslove this problem, Analytics-Zoo provides an end-to-end PPML platform for Big Data AI based on Intel SGX (Software Guard Extensions). This PPML platform ensure the whole Big Data & AI pipeline are fully protected by secured SGX enclave in hardware level, meanwhile exising Big Data & AI applications, such as Flink, Spark, SparkSQL and machine/deep learning, can be seamlessly migrated into this PPML platform without any code changes.

## PPML for Big Data AI

To take full advantage of big data, especially the value of private or sensitive data, customers need to build a trusted platform under the guidance of privacy laws or regulation, such as [GDPR](https://gdpr-info.eu/). This requirement raises big challenges to customers who already have big data and big data applications, such as Spark/SparkSQL, Flink/FlinkSQL and AI applications. Migrating these applications into privacy preserving way requires lots of additional efforts.

With Analytics-Zoo, customers can build a Trusted Platform for big data with a few clicks, and all existing big data & AI applications, such as Flink and Spark applications, can be migrated into this platform without any code changes. In specific, Analytics-Zoo uses serval security technologies

- Confiditional Computation with Intel SGX. Intel SGX provides hardware-based isolation and memory encryption with very limited attack surface.
- Seamless migration with LibOS. Based on LibOS projects ([Graphene](https://grapheneproject.io/) and [Occlum](https://occlum.io/)), Analytics-Zoo empowers our customers (e.g., data scientists and big data developers) to build PPML applications on top of large scale dataset without impacting existing applications.
- Secured networks with TLS and encryption. All network traffic are protected by TLS, in some cases, content should be encrypted before transformation.
- File or model protection with encryption. Model and senstive configuration files will be encrypted before uploading to Trusted platform. These files are only decrypted in SGX enclave. 
- Environment & App attestation with SGX attestation. SGX attestation ensure that remote/local SGX env and applications can be verified.


Note: Intel SGX requires hardware support, please [check if your CPU has this feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). In [3rd Gen Intel Xeon Scalable Processors](https://newsroom.intel.com/press-kits/3rd-gen-intel-xeon-scalable/), SGX allows up to 1TB of data to be included in secure enclaves.

### Key features

- Protecting data and model confidentiality
  - Sensitive input/output data (computation, training and inference), e.g., healthcare data
  - Propretary model, e.g., model trained with self-owned or sensitive data
- Seamless migrate existing big data applications into privacy preserving applications
- Trusted big data & AI Platform based on Intel SGX
  - Trusted Big Data Analytics and ML: Spark batch, SparkSQL, BigDL, TPC-H
  - Trusted Realtime Compute and ML: Flink batch/streaming, Cluster Serving

## Trusted Big Data Analytics and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Spark in SGX, then run Spark PI example in safe way. Trusted Big Data Analytics and ML supports Spark related applications, such as Spark batch jobs, SparkSQL and BigDL etc. For more examples, please refer to [trusted-big-data-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene).

### Use cases

- Big Data analysis using Spark (Spark SQL, Dataframe, MLlib, etc.)
- Distributed deep learning using BigDL

### Get started

#### Prerequisite: Install SGX Driver & Prepare Scripts

Please check if current platform [has SGX feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). Then, enable SGX feature in BIOS. Note that after SGX is enabled, a protion of memory will be assigned to SGX, and cannot be used or seen by OS.

Check SGX driver with `ls /dev | grep sgx`. If SGX driver is not installed, please install [SGX DCAP driver](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/driver/linux) with [install-graphene-driver.sh](https://github.com/intel-analytics/analytics-zoo/blob/master/ppml/scripts/install-graphene-driver.sh) (need root premission).

```bash
./ppml/scripts/install-graphene-driver.sh
```

#### Setp 0: Prepare Environment

Download scripts and dockerfiles in [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

##### TLS keys & password

Prepare keys for TLS (test only, need input security password for keys).

```bash
./ppml/scripts/generate-keys.sh
```

This scrips will generate 5 files in `keys` dir (you can replace them with your own TLS keys).

```bash
keystore.pkcs12
server.crt
server.csr
server.key
server.pem
```

Generated `password` to avoid plain text security password transfer.

```bash
./ppml/scripts/generate-password.sh
```

This scrips will generate 2 files in `password` dir.

```bash
key.txt
output.bin
```

##### Docker

Pull docker image from Dockerhub

```bash
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene:0.10-SNAPSHOT
```

Also, you can build docker image from Dockerfile (this will take some time).

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### Step 1: Start Trusted Big Data Analytics and ML Platform

Enter `analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene` dir. Start Spark service with this command

```bash
./start-local-big-data-ml.sh
sudo docker exec -it spark-local bash
```

##### Step 2: Run Spark Program

Modify `init.sh`, which is task/env related.

```bash
./init.sh
vim start-spark-local-pi-sgx.sh
```

Then run the script to run pi test in spark:

```bash
bash start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like:

>   Pi is roughly 3.1422957114785572

This example is a simple Spark local PI example, if you want to run application in distributed Spark cluster protected by SGX, please refer to [distributed bigdata ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene#start-the-distributed-bigdata-ml).

##### Step 3: Run BigDL Program

## Trusted Realtime Compute and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Flink in SGX, then run real-time model serving in safe way. Trusted Realtime Compute and ML supports Flink related applications, such as Batch/Streaming Flink jobs, FlinkSQL and Cluster Serving. For more examples, please refer to [trusted-realtime-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene).

### User cases

- Real time data computation/analytics using Flink
- Distributed end-to-end serving solution with Cluster Serving

### Get started

#### [Prerequisite: Install SGX Driver & Prepare Scripts](#prerequisite-install-sgx-driver--prepare-scripts)

#### Setp 0: Prepare Environment

Download scripts and dockerfiles in [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

##### TLS keys & password

Prepare keys for TLS (test only, need input security password for keys).

```bash
./ppml/scripts/generate-keys.sh
```

This scrips will generate 5 files in `keys` dir (you can replace them with your own TLS keys).

```bash
keystore.pkcs12
server.crt
server.csr
server.key
server.pem
```

Generated `password` to avoid plain text security password transfer.

```bash
./ppml/scripts/generate-password.sh
```

This scrips will generate 2 files in `password` dir.

```bash
key.txt
output.bin
```

##### Docker

Pull docker image from Dockerhub

```bash
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-occlum:0.10-SNAPSHOT
```

Also, you can build docker image from Dockerfile (this will take some time).

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### Step 1: Start  Trusted Realtime Compute and ML Platform

Enter `analytics-zoo/ppml/trusted-realtime-ml/scala/docker-graphene` dir.

Modify `environments.sh`. Change MASTER, WORKER IP and file paths.

```bash
nano environments.sh
```

Start Flink in SGX

```bash
./deploy-flink.sh
```

After all jobs are done, stop Flink in SGX

```bash
./stop-flink.sh
```

#### Step 2: Run Flink Program

Submit jobs to Flink

```bash
```

#### Step 3: Run Trusted Cluster Serving

[Analytics-Zoo Cluster serving](https://www.usenix.org/conference/opml20/presentation/song) is a distributed end-to-end inference service based on Flink. With SGX and TLS, this service can be fully secured for input data and model.

After all services are ready, then you can directly push inference requests int queue with [Restful API](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/#restful-api). Also, you can push image/input into queue with Python API

```python
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```

Cluster Serving service is a long running service in container, you can stop it with

```bash
docker stop containerID
```
