set -e

echo "#1 start k8s example test for anomaly_detection"
#timer
start=$(date "+%s")

${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
  --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.az=true \
  --conf spark.kubernetes.executor.label.az=true \
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
  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection.py \
  --input_dir file:///opt/nyc_taxi.csv --nb_epoch 1

now=$(date "+%s")
time1=$((now - start))

echo "#2 start k8s example test for pytorch estimator (test 'init_orca_context')"
start=$(date "+%s")

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/horovod/pytorch_estimator.py --cluster_mode k8s --k8s_master "k8s://https://127.0.0.1:8443" --container_image 10.239.47.32/intelanalytics/hyper-zoo:latest  --k8s_driver_host 172.168.3.101 --k8s_driver_port 54321

now=$(date "+%s")
time2=$((now - start))

echo "#1 anomaly_detection time used: $time1 seconds"
echo "#2 pytorch estimator time used: $time2 seconds"
