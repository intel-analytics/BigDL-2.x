# PPML (Privacy Preserving Machine Learning)

Analytics-Zoo provides an end-to-end PPML platform for Big Data AI.

## PPML for Big Data AI

To take full advantage of the value of big data, especially the value of private or sensitive data, customers need to build a trusted platform under the guidance of privacy laws or regulation, such as [GDPR](https://gdpr-info.eu/). This requirement raises a big challenge to customers who already have big data and big data applications, such as Spark/SparkSQL, Flink and AI applications. Migrating these applications into privacy preserving way requires lots of additional efforts.

To reslove this problem, Analytics-Zoo chooses [Intel SGX (Software Guard Extensions)](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html), a widely used [TEE (Trusted Execution Environment)](https://en.wikipedia.org/wiki/Trusted_execution_environment) technology, as main security building block for this PPML platforms. Different from other PPML technologies, e.g., [HE (Homomorphic Encryption)](https://en.wikipedia.org/wiki/Homomorphic_encryption), [MPC (Multi-Party Computation) or SMC (Secure Multi-Party Computation)](https://en.wikipedia.org/wiki/Secure_multi-party_computation) and [DP (Differential Privacy)](https://en.wikipedia.org/wiki/Differential_privacy), Intel SGX performs well on all measures (security, performance and utility).

<p align="center"><img src="../../../../../docs/Image/PPML/ppml_tech.png" height="180px"><br></p>

Based on Intel SGX (Software Guard Extensions) and LibOS projects ([Graphene](https://grapheneproject.io/) and [Occlum](https://occlum.io/)), Analytics-Zoo empowers our customers (e.g., data scientists and big data developers) to build PPML applications on top of large scale dataset without impacting existing applications.

![PPML Architecture](../../../../../docs/Image/PPML/ppml_arch.png#center)

Note: Intel SGX requires hardware support, please [check if your CPU has this feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). In [3rd Gen Intel Xeon Scalable Processors](https://newsroom.intel.com/press-kits/3rd-gen-intel-xeon-scalable/), SGX allows up to 1TB of data to be included in secure enclaves.

### Key features

- Protecting data and model confidentiality
- Seamless migrate existing big data applications into privacy preserving applications
- Trusted big data & AI Platform based on Intel SGX

### Scenario

- Protecting sensitive input/output data (computation, training and inference) in big data applications, e.g.,data analysis or machine learning on healthcare dataset
- Protecting propretary model in training and inference, e.g., secured model inference with self-owned model

## Trusted Big Data Analytics and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Spark in SGX, then run applications in safe way.

### Scenario

- Batch computation/analytics, i.e., privacy preserved Spark jobs
- Interactive computation/analytics, i.e., privacy preserved SparkSQL
- Large scale Spark related workload, e.g., TPC-H Benchmark
- Distributed machine learning & deep Learning with BigDL 

### Get started

#### Setp 0: Prepare Environment

Please clone or download Analytics-Zoo source code, then enter `analytics-zoo/ppml`.

If SGX driver is not installed, please install SGX driver with this command

```bash
./scripts/install-graphene-driver.sh
```

```bash
cd trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### Step 1: Start Spark in SGX

Enter `analytics-zoo/ppmltrusted-big-data-ml/scala/docker-graphene` dir.

```bash
./start-local-big-data-ml.sh
sudo docker exec -it spark-local bash
```

##### Step 2: Submit jobs to Spark

```bash
./init.sh
vim start-spark-local-pi-sgx.sh
```
Add these code in the `start-spark-local-pi-sgx.sh` file: <br>
```bash
#!/bin/bash

set -x

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

Then run the script to run pi test in spark: <br>
```bash
chmod a+x start-spark-local-pi-sgx.sh
./start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like: <br>
>   Pi is roughly 3.1422957114785572


For more examples, please refer to [trusted-big-data-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene).

## Trusted Realtime Compute and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Flink in SGX, then run real-time applications or model serving in safe way.

### Scenario

- Real time data computation/analytics, e.g., privacy preserved Flink jobs
- Privacy preserved distributed model inference with propretary model

## Get started

- Env setup (DockFIle)
- Flink example (word count)
- Cluster serving

```bash
```
