# K8s User Guide

---

### **1. Pull `hyper-zoo` Docker Image**

You may pull the prebuilt  Analytics Zoo `hyper-zoo` Image from [Docker Hub](https://hub.docker.com/r/intelanalytics/hyper-zoo/tags) as follows:

```bash
sudo docker pull intelanalytics/hyper-zoo:latest
```

**Speed up pulling image by adding mirrors**

To speed up pulling the image from DockerHub, you may add the registry-mirrors key and value by editing `daemon.json` (located in `/etc/docker/` folder on Linux):
```
{
  "registry-mirrors": ["https://<my-docker-mirror-host>"]
}
```
For instance, users in China may add the USTC mirror as follows:
```
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
```

After that, flush changes and restart docker：

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### **2. Launch a K8s Client Container**

Client container is for user to submit Analytics Zoo jobs from here, since it contains the required environment by Analytics Zoo.

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    intelanalytics/hyper-zoo:latest bash
```

**Note:** to launch the client container, `-v /etc/kubernetes:/etc/kubernetes:` and `-v /root/.kube:/root/.kube` are required to specify the path of kube config and installation.

To specify more arguments, use:

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -e NotebookPort=12345 \
    -e NotebookToken="your-token" \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=account \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/hyper-zoo:latest \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=myvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=4 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    intelanalytics/hyper-zoo:latest bash 
```

- NotebookPort value 12345 is a user specified port number.
- NotebookToken value "your-token" is a user specified string.
- http_proxy/https_proxy is to specify http proxy/https_proxy.
- RUNTIME_SPARK_MASTER is to specify spark master, which should be `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>` or `spark://<spark-master-host>:<spark-master-port>`. 
- RUNTIME_K8S_SERVICE_ACCOUNT is service account for driver pod. Please refer to k8s [RBAC](https://spark.apache.org/docs/latest/running-on-kubernetes.html#rbac).
- RUNTIME_K8S_SPARK_IMAGE is the k8s image.
- RUNTIME_PERSISTENT_VOLUME_CLAIM is to specify [Kubernetes volume](https://spark.apache.org/docs/latest/running-on-kubernetes.html#volume-mounts) mount. We are supposed to use volume mount to store or receive data.
- RUNTIME_DRIVER_HOST/RUNTIME_DRIVER_PORT is to specify driver localhost and port number (only required when submitting jobs via kubernetes client mode).
- Other environment variables are for spark configuration setting. The default values in this image are listed above. Replace the values as you need.

Once the container is created, launch the container by:

```bash
sudo docker exec -it <containerID> bash
```

You will login into the container and see this as the output:

```
root@[hostname]:/opt/spark/work-dir# 
```

`/opt/spark/work-dir` is the spark work path. 

The `/opt` directory contains:

- download-analytics-zoo.sh is used for downloading Analytics-Zoo distributions.
- start-notebook-spark.sh is used for starting the jupyter notebook on standard spark cluster. 
- start-notebook-k8s.sh is used for starting the jupyter notebook on k8s cluster.
- analytics-zoo-x.x-SNAPSHOT is `ANALYTICS_ZOO_HOME`, which is the home of Analytics Zoo distribution.
- analytics-zoo-examples directory contains downloaded python example code.
- jdk is the jdk home.
- spark is the spark home.
- redis is the redis home.

### **3. Run Analytics Zoo Examples on k8s**

_**Note**: Please make sure `kubectl` has appropriate permission to create, list and delete pod._

#### **3.1 Use `init_orca_context`**

- Client mode

We recommend using `init_orca_context` at the very beginning of your code to initiate and run Analytics Zoo on standard K8s clusters in [client mode](http://spark.apache.org/docs/latest/running-on-kubernetes.html#client-mode).

```python
from zoo.orca import init_orca_context

init_orca_context(cluster_mode="k8s", master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>",
                  container_image="intelanalytics/hyper-zoo:latest",
                  num_nodes=2, cores=2,
                  conf={"spark.driver.host": "x.x.x.x"})
```

- Cluster mode

For k8s [cluster mode](https://spark.apache.org/docs/2.4.5/running-on-kubernetes.html#cluster-mode), you can call init_orca_context and specify cluster_mode to be "spark-submit" in your python script:

```python
from zoo.orca import init_orca_context

init_orca_context(cluster_mode="spark-submit")
```

Use spark-submit to submit your Analytics Zoo program (e.g. script.py):

```bash
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
  --master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
  --deploy-mode cluster \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image="intelanalytics/hyper-zoo:latest" \
  --conf spark.executor.instances=1 \
  --executor-memory 10g \
  --driver-memory 10g \
  --executor-cores 8 \
  --num-executors 2 \
  file:///opt/script.py
```

#### **3.2 Use `spark_submit`**

Alternatively, you may use `spark_submit` to run your program on K8s clusters.

**Run Python programs**

```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip,/opt/analytics-zoo-examples/python/anomalydetection/anomaly_detection.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  file:///opt/analytics-zoo-examples/python/anomalydetection/anomaly_detection.py \
  --input_dir /zoo/data/nyc_taxi.csv
```
Above is a sample for submitting the python [anomalydetection](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/anomalydetection) example on client mode.

Options:

- --master: the spark mater, must be a URL with the format `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`. 
- --deploy-mode: submit application in client mode.
- --name: the Spark application name.
- --conf: require to specify k8s service account, container image to use for the Spark application, driver volumes name and path, label of pods, spark driver and executor configuration, etc.
  check the argument settings in your environment and refer to the [spark configuration page](https://spark.apache.org/docs/latest/configuration.html) and [spark on k8s configuration page](https://spark.apache.org/docs/latest/running-on-kubernetes.html#configuration) for more details.
- --properties-file: the customized conf properties.
- --py-files: the extra python packages is needed.
- file://: local file path of the python example file in the client container.
- --input_dir: input data path of the anomaly detection example. The data path is the mounted filesystem of the host. Refer to more details by [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes).

**Run Jupyter Notebooks**

After a Docker container is launched and user login into the container, you can start the Jupyter Notebook service inside the container.

In the `/opt` directory, run this command line to start the Jupyter Notebook service:
```
./start-notebook-k8s.sh
```

You will see the output message like below. This means the Jupyter Notebook service has started successfully within the container.
```
[I 01:04:45.625 NotebookApp] Serving notebooks from local directory: /opt/work/analytics-zoo-0.5.0-SNAPSHOT/apps
[I 01:04:45.625 NotebookApp] The Jupyter Notebook is running at:
[I 01:04:45.625 NotebookApp] http://(the-host-name or 127.0.0.1):12345/?token=...
[I 01:04:45.625 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Then, refer [docker guide](./docker.md) to open Jupyter Notebook service from a browser and run notebook.

**Run Scala programs**

Here is a sample for submitting the scala [anomalydetection](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/anomalydetection) example on cluster mode

```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection \
  ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --inputDir /zoo/data
```

Options:

- --master: the spark mater, must be a URL with the format `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`. 
- --deploy-mode: submit application in cluster mode.
- --name: the Spark application name.
- --conf: require to specify k8s service account, container image to use for the Spark application, driver volumes name and path, label of pods, spark driver and executor configuration, etc.
  check the argument settings in your environment and refer to the [spark configuration page](https://spark.apache.org/docs/latest/configuration.html) and [spark on k8s configuration page](https://spark.apache.org/docs/latest/running-on-kubernetes.html#configuration) for more details.
- --properties-file: the customized conf properties.
- --py-files: the extra python packages is needed.
- --class: scala example class name.
- --input_dir: input data path of the anomaly detection example. The data path is the mounted filesystem of the host. Refer to more details by [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes).

#### **3.3 Access logs and clear pods**

When application is running, it’s possible to stream logs on the driver pod:

```bash
$ kubectl logs <spark-driver-pod>
```

To check pod status or to get some basic information around pod using:

```bash
$ kubectl describe pod <spark-driver-pod>
```

You can also check other pods using the similar way.

After finishing running the application, deleting the driver pod:

```bash
$ kubectl delete <spark-driver-pod>
```

Or clean up the entire spark application by pod label:

```bash
$ kubectl delete pod -l <pod label>
```
