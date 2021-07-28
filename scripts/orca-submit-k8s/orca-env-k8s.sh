#env
NFS_SERVER="172.16.0.200"
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
PVC_NAME="persistent-volume-claim"
PVC_STORAGE="100Mi"
#put the user data and program here, they will be upload to shared storage to each spark driver and executors.
USER_LOCAL_STORAGE="/root/mydata"


MOUNT_PATH="/zoo"
RUNTIME_K8S_SERVICE_ACCOUNT="spark"
RUNTIME_EXECUTOR_INSTANCES="1"
RUNTIME_EXECUTOR_CORES="4"
RUNTIME_EXECUTOR_MEMORY="20g"
RUNTIME_TOTAL_EXECUTOR_CORES="4"
RUNTIME_DRIVER_CORES="4"
RUNTIME_DRIVER_MEMORY="10g"

NFS_CLIENT_PROVISIONER_IMAGE="10.239.45.10/arda/external_storage/nfs-client-provisioner:latest"
SPARK_K8S_MASTER="k8s://https://127.0.0.1:8443"
DRIVER_HOST="172.16.0.200"
DRIVER_PORT="54321"
AZ_CLIENT_NAME="azclient"
K8S_CONFIG="/root/mydata/kuberconfig"
SPARK_K8S_IMAGE="10.239.45.10/arda/hyper-zoo:e2c"

#USER_COMMEND="cd /zoo/data/e2c-analytics-stack/ref_algos/inspection/train && python train_az_k8s_client.py"
USER_COMMEND="/opt/analytics-zoo-0.10.0-SNAPSHOT/bin/spark-submit-python-with-zoo.sh \
--master $SPARK_K8S_MASTER \
--deploy-mode cluster \
--name e2c \
--conf spark.kubernetes.container.image="$SPARK_K8S_IMAGE" \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="$PVC_NAME" \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="$MOUNT_PATH" \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="$PVC_NAME" \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="$MOUNT_PATH" \
--conf spark.executor.instances=4 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=$RUNTIME_K8S_SERVICE_ACCOUNT \
--executor-memory 50g \
--driver-memory 50g \
--py-files "/zoo/data/e2c-analytics-stack/ref_algos/inspection/train/model.py" \
--executor-cores 8 \
--num-executors 4 \
--total-executor-cores 32 \
file:///zoo/data/e2c-analytics-stack/ref_algos/inspection/train/train_az_k8s_cluster.py"

