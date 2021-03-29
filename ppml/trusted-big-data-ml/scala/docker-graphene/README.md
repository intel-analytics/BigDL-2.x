# Trusted Big Data ML
SGX-based Trusted Big Data ML allows user to run end to end big data analytics application and Intel Analytics Zoo and BigDL model training with spark local and distributed cluster on Graphene-SGX.

*Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.*


## How To Build 
Before run the following command, please modify the pathes in the build-docker-image.sh file at first. <br>
Then build docker image by running this command: <br>
```bash
./build-docker-image.sh
```

## How to Run

### Prerequisite
To launch Trusted Big Data ML applications on Graphene-SGX, you need to install graphene-sgx-driver:
```bash
../../../scripts/install-graphene-driver.sh
```

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
There're four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>

### Prepare the keys
The ppml in analytics zoo needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores.
This script is in /analytics-zoo/ppml/scripts:
```bash
../../../scripts/generate-keys.sh
```
### Prepare the password
You also need to store the password you used in previous step in a secured file:
This script is also in /analytics-zoo/ppml/scripts:
```bash
../../../scripts/generate-password.sh used_password_when_generate_keys
```

### Run the PPML as Docker containers

#### In spark local mode
##### Start the container to run spark applications in ppml
Before you run the following command to start container, you need to modify the paths in the start-local-big-data-ml.sh. <br>
Then run the following command: <br>
```bash
./start-local-big-data-ml.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
```

##### Example 1: Spark PI on Graphene-SGX
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

##### Example 2: Analytics Zoo model training on Graphene-SGX
```bash
./init.sh
./start-spark-local-train-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.sgx.log | egrep "###|INFO"
```
or
```bash
sudo docker logs spark-local | egrep "###|INFO"
```

The result should look like: <br>
>   ############# train optimized[P1182:T2:java] ---- end time: 310534 ms return from shim_write(...) = 0x1d <br>
>   ############# ModuleLoader.saveToFile File.saveBytes end, used 827002 ms[P1182:T2:java] ---- end time: 1142754 ms return from shim_write(...) = 0x48 <br>
>   ############# ModuleLoader.saveToFile saveWeightsToFile end, used 842543 ms[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x4b <br>
>   ############# model saved[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x19 <br>

##### Example 3: Spark TPC-H on Graphene-SGX
Before run TPC-H test in container we created, we should download and install [SBT](https://www.scala-sbt.org/download.html), then build and package TPC-H dataset according to [TPC-H](https://github.com/qiuxin2012/tpch-spark) with your needs. After packaged, check if we have `spark-tpc-h-queries_2.11-1.0.jar ` under `/tpch-spark/target/scala-2.11`, if have, we package successfully.

Copy TPC-H to container: <br>
```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml/work
sudo docker exec -it spark-local bash
cd ppml/trusted-big-data-ml/
./init.sh
vim start-spark-local-tpc-h-sgx.sh
```

Add these code in the `start-spark-local-tpc-h-sgx.sh` file: <br>
```bash
#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar:/ppml/trusted-big-data-ml/work/tpch-spark/dbgen/*:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
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
        --class main.scala.TpchQuery \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar \
        /ppml/trusted-big-data-ml/work/tpch-spark/dbgen | tee spark.local.tpc.h.sgx.log
```

Then run the script to run TPC-H test in spark: <br>
```bash
chmod a+x start-spark-local-tpc-h-sgx.sh
./start-spark-local-tpc-h-sgx.sh
```

Open another terminal and check the log: <br>
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO"
```
##### Other Spark workloads are also supported, please follow the 3 examples to submit your workload with spark on Graphene-SGX


#### In spark standalone cluster mode
##### setup passwordless ssh login to all the nodes.
##### config the environments for master, workers, docker image, security keys/passowrd files and data path.
```bash
nano environments.sh
```
##### start the distributed bigdata ml
```bash
./start-distributed-big-data-ml.sh
```
##### stop the distributed bigdata ml
```bash
./stop-distributed-big-data-ml.sh
```
##### Other Spark workloads are also supported, please follow the 3 examples to submit your workload with spark on Graphene-SGX
