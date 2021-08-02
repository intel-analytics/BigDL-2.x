source orca-env.sh

kubectl delete -f deploy-nfs/deployment.yaml
kubectl delete -f deploy-nfs/class.yaml
kubectl describe pvc $PVC_NAME | grep Finalizers
kubectl patch pvc $PVC_NAME -p '{"metadata":{"finalizers": []}}' --type=merge
kubectl delete -f deploy-nfs/persistent-volume-claim.yaml
