#!/bin/bash

export ENCLAVE_KEY_PATH=/home/sdp/qiyuan/redis_test/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem
export DATA_PATH=~/hangrui/myzoo/analytics-zoo/ppml/trusted-big-data-ml/work/data/
export KEYS_PATH=/home/sdp/qiyuan/keys
export LOCAL_IP=192.168.0.113

sudo docker pull 10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.11-SNAPSHOTCHR

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-3" \
    --oom-kill-disable \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /dev/sgx/enclave:/dev/sgx/enclave \
    -v /dev/sgx/provision:/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local-new \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
     10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.11-SNAPSHOTCHR
