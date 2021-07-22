#env
NFS_SERVER_VALUE="172.16.0.200"
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
NFS_CLIENT_PROVISIONER_IMAGE="10.239.45.10/arda/external_storage/nfs-client-provisioner:latest"
PVC_NAME="persistent-volume-claim"
PVC_STORAGE="100Mi"
SPARK_K8S_MASTER="k8s://https://127.0.0.1:8443"
DRIVER_HOST="172.16.0.200"
DRIVER_PORT="54321"
AZ_CLIENT_NAME="azclient11"
K8S_CONFIG="/root/glorysdj/kuberconfig"
IMAGE="10.239.45.10/arda/hyper-zoo:e2c"
USER_FOLDER="/root/glorysdj"

COMMEND="cd /zoo/data/e2c-analytics-stack/ref_algos/inspection/train && python train_az_k8s_client.py"
COMMEND1="/opt/analytics-zoo-0.10.0-SNAPSHOT/bin/spark-submit-python-with-zoo.sh \
--master $SPARK_K8S_MASTER \
--deploy-mode cluster \
--name e2c \
--conf spark.kubernetes.container.image="$IMAGE" \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="$PVC_NAME" \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="/zoo" \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="$PVC_NAME" \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="/zoo" \
--conf spark.executor.instances=4 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--executor-memory 50g \
--driver-memory 50g \
--py-files "/zoo/data/e2c-analytics-stack/ref_algos/inspection/train/model.py" \
--executor-cores 8 \
--num-executors 4 \
--total-executor-cores 32 \
file:///zoo/data/e2c-analytics-stack/ref_algos/inspection/train/train_az_k8s_cluster.py"

#check nfs server
showmount -e $NFS_SERVER_VALUE
var=`showmount -e $NFS_SERVER_VALUE`
NFS_SHARE_DATA_PATH="/disk1/nfsdata"
result=$(echo $var | grep "${NFS_SHARE_DATA_PATH}")
if [[ "$result" != "" ]]
then
    echo "nfs success"
else
    echo "nfs failed"
fi

#deploy pvc
sed -n "s/TMP_NFS_SERVER_VALUE/$NFS_SERVER_VALUE/g" deploy-nfs/deployment.yaml.template | tee deploy-nfs/deployment.yaml
sed -i "s#TMP_NFS_SHARE_DATA_PATH#$NFS_SHARE_DATA_PATH#g" deploy-nfs/deployment.yaml
sed -i "s#TMP_NFS_CLIENT_PROVISIONER_IMAGE#$NFS_CLIENT_PROVISIONER_IMAGE#g" deploy-nfs/deployment.yaml
sed -n "s/TMP_PVC_NAME/$PVC_NAME/g" deploy-nfs/persistent-volume-claim.yaml.template | tee deploy-nfs/persistent-volume-claim.yaml
sed -i "s/TMP_PVC_STORAGE/$PVC_STORAGE/g" deploy-nfs/persistent-volume-claim.yaml

kubectl create -f deploy-nfs/deployment.yaml
kubectl create -f deploy-nfs/class.yaml
kubectl create -f deploy-nfs/persistent-volume-claim.yaml

#copy to nfsdata
var1=`ls $NFS_SHARE_DATA_PATH | grep "$PVC_NAME" | grep -v "archived"`
echo $var1
cp -r $USER_FOLDER/data $NFS_SHARE_DATA_PATH/$var1
mkdir $NFS_SHARE_DATA_PATH/$var1/logs

#run image
docker pull $IMAGE
docker run -itd --net=host --name $AZ_CLIENT_NAME \
    -v $K8S_CONFIG:/root/.kube/config \
    -v $USER_FOLDER:/zoo \
    -e RUNTIME_SPARK_MASTER=$SPARK_K8S_MASTER \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=$IMAGE \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=$PVC_NAME \
    -e RUNTIME_DRIVER_HOST=$DRIVER_HOST \
    -e RUNTIME_DRIVER_PORT=$DRIVER_PORT \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=4 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    $IMAGE bash
docker exec -it $AZ_CLIENT_NAME bash -c "cd /zoo && $COMMEND"

