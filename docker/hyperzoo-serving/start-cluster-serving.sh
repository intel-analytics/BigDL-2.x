# start cluster-serving
${FLINK_HOME}/bin/flink run-application \
  -c com.intel.analytics.zoo.serving.ClusterServing \
  -m  ${FLINK_UI_HOST}:${FLINK_UI_PORT} \
  --target kubernetes-application \
  -Dkubernetes.cluster-id=${RUNTIME_K8S_FLINK_CLUSER_ID} \
  -Dkubernetes.container.image=${RUNTIME_K8S_FLINK_IMAGE} \
  -Dkubernetes.service-account=${RUNTIME_K8S_FLINK_SERVICE_ACCOUNT} \
  -Dkubernetes.namespace=${RUNTIME_K8S_FLINK_NAME_SPACE} \
  -Djobmanager.memory.process.size=${JOBMANAGER_MEMORY_PROCESS_SIZE} \
  -Dtaskmanager.memory.process.size=${TASKMANAGER_MEMORY_PROCESS_SIZE} \
  -Dtaskmanager.memory.task.off-heap.size=${TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE} \
  -Dtaskmanager.memory.framework.off-heap.size=${TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE} \
  local:///opt/work/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-serving.jar \
  --configPath ${CONFIG_DIR}/serving-conf.yaml
