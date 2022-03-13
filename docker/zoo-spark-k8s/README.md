# zoo-spark-k8s
It is stated in advance that this module is only for tasks submitted in `client` mode.

In order to reduce the size of the image, **zoo-spark-k8s** builds an image without python dependencies, and pulls user-defined python environments remotely to provide environment dependencies for execute tasks.

* [Build Image](#BuildImage)
* [Start Container](#StartContainer)
* [Python Environment](#PythonEnvironment)
	* [Prerequirement](#Prerequirement)
	* [Package and Upload](#PackageandUpload)
* [Submit Job](#SubmitJob)


##  1. <a name='BuildImage'></a>Build Image
Use the Dockerfile provided by this module to build the image.

Need to specify the value of the `JDK_VERSION` and `JDK_URL` parameters.

```shell
docker build \
  --build-arg http_proxy=${HTTP_PROXY} \
  --build-arg https_proxy=${HTTPS_PROXY} \
  --build-arg JDK_VERSION=${JDK_VERSION} \
  --build-arg JDK_URL=${JDK_URL} \
  --build-arg no_proxy=${PROXY_IP} \
  --rm -t <IMAGE_TAG> -f Dockerfile .
```

##  2. <a name='StartContainer'></a>Start Container
Since the container contains the `analytics-zoo` related environment, start the container as the environment for submitting tasks.

`${RUNTIME_K8S_SPARK_IMAGE}` is the image built in the previous step.

```shell
docker run -itd --net=host \
    --name=${CONTAINER_NAME} \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v ${KUBE_CONFIG}:${KUBE_CONFIG} \
    -e NotebookPort=${NotebookPort} \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e NotebookToken=${NotebookToken} \
    -e RUNTIME_SPARK_MASTER=${MASTER_URL} \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=${RUNTIME_K8S_SPARK_IMAGE} \
    -e RUNTIME_DRIVER_HOST=${RUNTIME_DRIVER_HOST} \
    -e RUNTIME_DRIVER_PORT=${RUNTIME_DRIVER_PORT} \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=16 \
    -e RUNTIME_EXECUTOR_MEMORY=80g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=64 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=80g \
${RUNTIME_K8S_SPARK_IMAGE} bash
```

##  3. <a name='PythonEnvironment'></a>Python Environment
The key to this step is to create a python virtual environment in the container started in the previous step, and then install the required python package into the virtual environment.

Use the `conda pack` command to pack the installed environment and upload it to the user's own remote http server.

###  3.1. <a name='Prerequirement'></a>Prerequirement
* `requirements.txt` file: export the required python dependencies to the requirements.txt file
* Require an **http server** can upload files

###  3.2. <a name='PackageandUpload'></a>Package and Upload
The `install-zoo-conda.sh` script is responsible for using conda to package and upload the python environment,The user needs to modify the `install-zoo-conda.sh` script when using it, and assign values to the variables in the script.

* `PYTHON_VERSION`: Python version. For example: "3.6.9"
* `CONDA_ENV_NAME`: Custom compressed package name. For example: "zoo-conda"
* `MINICONDA`: Conda version. For example: "Miniconda3-py37_4.10.3-Linux-x86_64.sh"
* `REQUIREMENTS`: The requirements.txt file path. For example: /opt/spark/work-dir/requirements.txt
* `CURL_TOKEN`: curl `--user` option value, For example: "user:passwd"
* `PYTHON_DEPS_PATH`: File path after packaging, For example: "/opt/spark/work-dir/zoo-conda.tar.gz"
* `UPLOAD_URL`: The upload path of python environment, For example: "http://ip:port/repository/remote-storage/zoo-conda.tar.gz"
```shell
./install-zoo-conda.sh
```

##  4. <a name='SubmitJob'></a>Submit Job
The first step: `source activate ${CONDA_ENV_NAME}`.

The following is an example of submitting a task,there are some parameters that need to be customized:
* `${JOB_NAME}`: Job name, For example: analytics-zoo
* `${DRIVER_HOST}`: Driver host
* `${DRIVER_PORT}`: Driver port
* `${ZOO_CONDA_REMOTE_URL}`: The link of the uploaded python environment
* `${REMOTE_PYTHON_FILE_URL}`: Remote python file url
* `${REMOTE_INPUT_DATA_URL}`: Remote data url, Cannot be a directory

```shell
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name ${JOB_NAME} \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.label.az=true \
  --conf spark.kubernetes.executor.label.az=true \
  --conf spark.kubernetes.node.selector.spark=true \
  --conf spark.driver.host=${DRIVER_HOST} \
  --conf spark.driver.port=${DRIVER_PORT} \
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
  --files ${ZOO_CONDA_REMOTE_URL} \
  --conf spark.pyspark.driver.python=python \
  --conf spark.pyspark.python=./bin/python \
  ${REMOTE_PYTHON_FILE_URL} \
  --input_dir ${REMOTE_INPUT_DATA_URL}
```

