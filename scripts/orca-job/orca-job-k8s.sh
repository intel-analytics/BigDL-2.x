
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
sed "s/TMP_NFS_SERVER_VALUE/$NFS_SERVER/g" $NFS_DEPLOY_PATH/deployment.yaml.template | tee $NFS_DEPLOY_PATH/deployment.yaml
sed -i "s#TMP_NFS_SHARE_DATA_PATH#$NFS_SHARE_DATA_PATH#g" $NFS_DEPLOY_PATH/deployment.yaml
sed -i "s#TMP_NFS_CLIENT_PROVISIONER_IMAGE#$NFS_CLIENT_PROVISIONER_IMAGE#g" $NFS_DEPLOY_PATH/deployment.yaml
sed "s/TMP_PVC_NAME/$PVC_NAME/g" $NFS_DEPLOY_PATH/persistent-volume-claim.yaml.template | tee $NFS_DEPLOY_PATH/persistent-volume-claim.yaml
sed -i "s/TMP_PVC_STORAGE/$PVC_STORAGE/g" $NFS_DEPLOY_PATH/persistent-volume-claim.yaml

kubectl create -f $NFS_DEPLOY_PATH/deployment.yaml
kubectl create -f $NFS_DEPLOY_PATH/class.yaml
kubectl create -f $NFS_DEPLOY_PATH/persistent-volume-claim.yaml

kubectl get sc
kubectl get pv

#copy to nfsdata
var1=`ls $NFS_SHARE_DATA_PATH | grep "$PVC_NAME" | grep -v "archived"`
echo $var1
#cp -r $USER_LOCAL_STORAGE/data $NFS_SHARE_DATA_PATH/$var1

sed "s#USER_COMMAND#$USER_COMMAND#g" orca-job.yaml.template | tee orca-job.yaml
sed -i "s#SPARK_K8S_IMAGE#$SPARK_K8S_IMAGE#g" orca-job.yaml
sed -i "s#PVC_NAME#$PVC_NAME#g" orca-job.yaml
sed -i "s#MOUNT_PATH#$MOUNT_PATH#g" orca-job.yaml

#run client container
kubectl delete -f orca-job.yaml
kubectl create -f orca-job.yaml
kubectl get pod | grep job

