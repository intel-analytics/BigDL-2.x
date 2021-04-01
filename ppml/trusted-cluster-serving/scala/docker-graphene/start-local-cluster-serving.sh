export KEYS_PATH=the_dir_path_of_your_prepared_keys
export SECURE_PASSWORD_PATH=the_dir_path_of_your_prepared_password
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-30" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $KEYS_PATH:/ppml/trusted-cluster-serving/redis/work/keys \
    -v $KEYS_PATH:/ppml/trusted-cluster-serving/java/work/keys \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-cluster-serving/redis/work/passowrd \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-cluster-serving/java/work/passowrd \
    --name=flink-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e CORE_NUM=30 \
    intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT \
    bash  -c "cd /ppml/trusted-cluster-serving/ && ./start-all.sh"
