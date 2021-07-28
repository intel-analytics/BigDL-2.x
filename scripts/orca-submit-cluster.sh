#env
NFS_SERVER="172.16.0.200"
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
PVC_NAME="persistent-volume-claim"
PVC_STORAGE="100Mi"
#put the user data and program here, they will be upload to shared storage to each spark driver and executors.
USER_LOCAL_STORAGE="/root/glorysdj"
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
AZ_CLIENT_NAME="azclient12"
K8S_CONFIG="/root/glorysdj/kuberconfig"
SPARK_K8S_IMAGE="10.239.45.10/arda/hyper-zoo:e2c"
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

#check nfs server
showmount -e $NFS_SERVER
var=`showmount -e $NFS_SERVER`
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
result=$(echo $var | grep "${NFS_SHARE_DATA_PATH}")
if [[ "$result" != "" ]]
then
    echo "nfs success"
else
    echo "nfs failed"
fi

#deploy pvc
sed "s/TMP_NFS_SERVER_VALUE/$NFS_SERVER/g" deploy-nfs/deployment.yaml.template | tee deploy-nfs/deployment.yaml
sed -i "s#TMP_NFS_SHARE_DATA_PATH#$NFS_SHARE_DATA_PATH#g" deploy-nfs/deployment.yaml
sed -i "s#TMP_NFS_CLIENT_PROVISIONER_IMAGE#$NFS_CLIENT_PROVISIONER_IMAGE#g" deploy-nfs/deployment.yaml
sed "s/TMP_PVC_NAME/$PVC_NAME/g" deploy-nfs/persistent-volume-claim.yaml.template | tee deploy-nfs/persistent-volume-claim.yaml
sed -i "s/TMP_PVC_STORAGE/$PVC_STORAGE/g" deploy-nfs/persistent-volume-claim.yaml

kubectl create -f deploy-nfs/deployment.yaml
kubectl create -f deploy-nfs/class.yaml
kubectl create -f deploy-nfs/persistent-volume-claim.yaml

#copy to nfsdata
var1=`ls $NFS_SHARE_DATA_PATH | grep "$PVC_NAME" | grep -v "archived"`
echo $var1
cp -r $USER_LOCAL_STORAGE/data $NFS_SHARE_DATA_PATH/$var1
mkdir $NFS_SHARE_DATA_PATH/$var1/logs

#run client container
docker pull $SPARK_K8S_IMAGE
docker run -itd --net=host --name $AZ_CLIENT_NAME \
    -v $K8S_CONFIG:/root/.kube/config \
    -v $USER_LOCAL_STORAGE:$MOUNT_PATH \
    -e RUNTIME_SPARK_MASTER=$SPARK_K8S_MASTER \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=$RUNTIME_K8S_SERVICE_ACCOUNT \
    -e RUNTIME_K8S_SPARK_IMAGE=$SPARK_K8S_IMAGE \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=$PVC_NAME \
    -e RUNTIME_DRIVER_HOST=$DRIVER_HOST \
    -e RUNTIME_DRIVER_PORT=$DRIVER_PORT \
    -e RUNTIME_EXECUTOR_INSTANCES=$RUNTIME_EXECUTOR_INSTANCES \
    -e RUNTIME_EXECUTOR_CORES=$RUNTIME_EXECUTOR_CORES \
    -e RUNTIME_EXECUTOR_MEMORY=$RUNTIME_EXECUTOR_MEMORY \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=$RUNTIME_TOTAL_EXECUTOR_CORES \
    -e RUNTIME_DRIVER_CORES=$RUNTIME_DRIVER_CORES \
    -e RUNTIME_DRIVER_MEMORY=$RUNTIME_DRIVER_MEMORY \
    $SPARK_K8S_IMAGE bash
docker exec -it $AZ_CLIENT_NAME bash -c "cd $MOUNT_PATH && $USER_COMMEND"
