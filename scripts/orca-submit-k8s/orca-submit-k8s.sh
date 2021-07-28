source orca-env.sh

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
