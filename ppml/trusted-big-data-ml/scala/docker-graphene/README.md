# trusted-big-data-ml
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.


## How To Build 

```bash
export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export JDK_URL=http://your-http-url-to-download-jdk
sudo docker build \
    --build-arg http_proxy=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT \
    --build-arg https_proxy=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT \
    --build-arg HTTP_PROXY_HOST=$HTTP_PROXY_HOST \
    --build-arg HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
    --build-arg HTTPS_PROXY_HOST=$HTTPS_PROXY_HOST \
    --build-arg HTTPS_PROXY_PORT=$HTTPS_PROXY_PORT \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=$JDK_URL \
    --build-arg no_proxy=x.x.x.x \
    -t analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT -f ./Dockerfile .
```

## How to Run

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
There're four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>

### Prepare the keys
The ppml in analytics zoo need secured keys to enable spark security such as AUTHENTICATION, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores.
```bash
    mkdir keys && cd keys
    openssl genrsa -des3 -out server.key 2048
    openssl req -new -key server.key -out server.csr
    openssl x509 -req -days 9999 -in server.csr -signkey server.key -out server.crt
    cat server.key > server.pem
    cat server.crt >> server.pem
    openssl pkcs12 -export -in server.pem -out keystore.pkcs12
    keytool -importkeystore -srckeystore keystore.pkcs12 -destkeystore keystore.jks -srcstoretype PKCS12 -deststoretype JKS
    openssl pkcs12 -in keystore.pkcs12 -nodes -out server.pem
    openssl rsa -in server.pem -out server.key
    openssl x509 -in server.pem -out server.crt
```

### Run the PPML Docker image

#### In spark local mode
##### Start the container to run SparkPi test in ppml
```bash
export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest \
    bash
sudo docker exec -it spark-local bash
cd ppml/trusted-bid-data-ml
export SPARK_HOME=/ppml/trusted-big-data-ml/work/spark-2.3.4
export SCALA_HOME=/ppml/trusted-big-data-ml/work/scala-2.11.12
export PATH=$SPARK_HOME/bin:$PATH:$SCALA_HOME
./init.sh
$SPARK_HOME/bin/spark-submit --class org.apache.spark.examples.SparkPi --master local $SPARK_HOME/examples/jars/spark-examples_2.11-2.4.3.jar
```
The result shows like this: <br>
>   Pi is roughly 3.1384956924784624

##### Start the container to run lenet model training with BigDL in ppml.
```bash
export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest \
    bash -c "cd /ppml/trusted-big-data-ml/ && ./init.sh && ./start-spark-local-train-sgx.sh"
```
check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.sgx.log | egrep "###|INFO"
```
or
```bash
sudo docker logs spark-local | egrep "###|INFO"
```
The result shows like: <br>
>   ############# train optimized[P1182:T2:java] ---- end time: 310534 ms return from shim_write(...) = 0x1d <br>
>   ############# ModuleLoader.saveToFile File.saveBytes end, used 827002 ms[P1182:T2:java] ---- end time: 1142754 ms return from shim_write(...) = 0x48 <br>
>   ############# ModuleLoader.saveToFile saveWeightsToFile end, used 842543 ms[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x4b <br>
>   ############# model saved[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x19 <br>

##### Start the container to run TPC-H in ppml.
Download SBT: <br>
```bash
wget https://github.com/sbt/sbt/releases/download/v1.4.8/sbt-1.4.8.tgz
tar -zxvf sbt-1.4.8.tgz
wget https://archive.apache.org/dist/spark/spark-2.3.4/spark-2.3.4.tgz
tar -zxvf spark-2.3.4.tgz
wget https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz
tar -zxvf scala-2.11.12.tgz
export SBT_HOME=the_dir_path_of_your_unzipped_sbt-1.4.8
export SPARK_HOME=the_dir_path_of_your_unzipped_spark-2.3.4
export SCALA_HOME=the_dir_path_of_your_unzipped_scala-2.11.12
export PATH=$SBT_HOME:$SPARK_HOME/bin:$PATH:$SCALA_HOME
cd sbt
vim ./sbt
```
Please add these into this sbt file and save it:
```bash
SBT_OPTS="-Xms512M -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256M"
java $SBT_OPTS -jar $SBT_HOME/bin/sbt-launch.jar "$@"
```
If there is some mistake, try to change your $SBT_HOME with the absulate path

Build SBT: <br>
```bash
chmod u+x ./sbt
./sbt sbtVersion
```
Run `./sbt sbtVersion` the first time, the machine would take some time to download some dependecies.

Download and build TPC-H: <br>
```bash
git clone https://github.com/qiuxin2012/tpch-spark.git
cd tpch-spark/dbgen
make
./dbgen -h
./dbgen
```
The result shows like: <br>
>   TPC-H Population Generator (Version 2.17.0)
>   Copyright Transaction Processing Performance Council 1994 - 2010

Build the TPC-H with 10G: <br>
```bash
./dbgen -s 10
```

Package TPC-H with SBT: <br>
```bash
cd ..
sbt package
```
Check if we have `spark-tpc-h-queries_2.11-1.0.jar ` under `/tpch-spark/target/scala-2.11`, if have, we package successfully.

Create the container: <br>
```bash
export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest \
    bash
```
Copy TPC-H to container: <br>
```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml
sudo docker exec -it spark-local bash
cd ppml/trusted-big-data-ml/tpch-spark
export PATH=$SPARK_HOME/bin:$PATH
spark-submit --class "main.scala.TpchQuery" --master local[*] target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar dbgen
```
The result shows like this: <br>
>   INFO Executor: Finished task 6.0 in stage 286.0 (TID 25381). 7928 bytes result sent to driver
>   INFO TaskSetManager: Finished task 6.0 in stage 286.0 (TID 25381) in 13 ms on localhost (executor driver) (7/7)
>   INFO TaskSchedulerImpl: Removed TaskSet 286.0, whose tasks have all completed, from pool
>   INFO DAGScheduler: ResultStage 286 (save at TpchQuery.scala:42) finished in 0.040 s
>   INFO DAGScheduler: Job 37 finished: save at TpchQuery.scala:42, took 0.226678 s
>   INFO FileFormatWriter: Write Job badb70fd-2b10-47ef-b58a-f1bfc3e026ca committed.
>   INFO FileFormatWriter: Finished processing stats for write job badb70fd-2b10-47ef-b58a-f1bfc3e026ca.

#### In spark standalone cluster mode
