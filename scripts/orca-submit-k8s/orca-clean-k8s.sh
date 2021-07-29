source orca-env.sh

kubectl delete -f $NFS_DEPLOY_PATH/deployment.yaml
kubectl delete -f $NFS_DEPLOY_PATH/class.yaml
kubectl describe pvc $PVC_NAME | grep Finalizers
kubectl patch pvc $PVC_NAME -p '{"metadata":{"finalizers": []}}' --type=merge
kubectl delete -f $NFS_DEPLOY_PATH/persistent-volume-claim.yaml
