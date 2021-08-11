# README

Please prepare your data and program in a folder. Then, modify the environment variables in `orca-env-k8s.sh` . You can use the `DEPLOY_MODE` variable to determine  client mode or cluster mode.  Run `orca-job-k8s.sh` to create the job. You may use `orca-clean-k8s.sh` to clean  pvc and delete related pods.  In this example, the program is  `spark_pandas.py` and the data is `nyc_taxi.csv` .



The `orca-env-k8s.sh` contains the environment variables, you should modify this file before run `orca-job-k8s.sh`. 

There are some variables you need to pay attention.

| Environment variables | Description                                 |
| --------------------- | ------------------------------------------- |
| NFS_SERVER            | The server IP of NFS                        |
| NFS_SHARE_DATA_PATH   | NFS's share data path                       |
| PVC_NAME              | The name of persistent-volume-claim         |
| PVC_STORAGE           | The storage memory of pvc                   |
| USER_LOCAL_STORAGE    | The path of user data and program           |
| K8S_CONFIG            | The path of k8s config file                 |
| DEPLOY_MODE           | Deploy mode includes "client" and "cluster" |
| USER_COMMEND          | The command used to run program             |

The `deploy-nfs` folder contains `deployment.template.yaml` , `class.yaml`  and `persistent-volume-claim.template.yaml` . `deployment.template.yaml` is used to create nfs client provisioner. The `class.yaml` could create storage class for nfs. The `persistent-volume-claim.template.yaml` could create a volume to store data. The nfs would be deployed by `orca-submit-k8s.sh` .



The `orca-job-k8s.sh` includes four phases. First, it would check whether nfs server is success or not. The next phase is use files in `deploy-nfs` to deploy persistent volume claim. Then, it would copy user data to nfs share data folder. After that, it would create a job to execute your command.



You may use `orca-clean-k8s.sh` to delete job and nfs. It would clean job, persistent volume claim, nfs client provisioner and storage class. 

