#env
NFS_SERVER="172.16.0.200"
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
NFS_DEPLOY_PATH="deploy-nfs"
PVC_NAME="persistent-volume-claim"
PVC_STORAGE="100Mi"
#put the user data and program here, they will be upload to shared storage to each spark driver and executors.
USER_LOCAL_STORAGE="/root/mydata"

MOUNT_PATH="/mnt"
RUNTIME_K8S_SERVICE_ACCOUNT="spark"

NFS_CLIENT_PROVISIONER_IMAGE="10.239.45.10/arda/external_storage/nfs-client-provisioner:latest"
SPARK_K8S_MASTER="k8s://https://127.0.0.1:8443"
DRIVER_HOST="172.16.0.200"
DRIVER_PORT="54321"
AZ_CLIENT_NAME="azclient"
K8S_CONFIG="/root/mydata/kuberconfig"
SPARK_K8S_IMAGE="10.239.45.10/arda/intelanalytics/hyper-zoo:0.11.0"
K8S_MASTER="k8s://https://127.0.0.1:8443"
#DEPLOY_MODE="cluster"
DEPLOY_MODE="client"
EXECUTOR_MEMORY="50g"
DIRVER_MEMORY="50g"
EXECUTOR_CORES="8"
NUM_EXECUTORS="4"
TOTAL_CORES="32"

USER_COMMAND="/opt/analytics-zoo-0.11.0/bin/spark-submit-python-with-zoo.sh \
--master $K8S_MASTER \
--deploy-mode $DEPLOY_MODE \
--name hyperzoo \
--conf spark.kubernetes.container.image=''SPARK_K8S_IMAGE'' \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=''$PVC_NAME'' \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=''$MOUNT_PATH'' \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=''$PVC_NAME'' \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=''$MOUNT_PATH'' \
--conf spark.executor.instances=4 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=$RUNTIME_K8S_SERVICE_ACCOUNT \
--executor-memory $EXECUTOR_MEMORY \
--driver-memory $DIRVER_MEMORY \
--executor-cores $EXECUTOR_CORES \
--num-executors $NUM_EXECUTORS \
--total-executor-cores $TOTAL_CORES \
file:///opt/analytics-zoo-examples/python/orca/data/spark_pandas.py \
-f /mnt/data/nyc_taxi.csv"
