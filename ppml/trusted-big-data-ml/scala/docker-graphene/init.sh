#!/bin/bash

set -x

if [ $# -ne 2 ]; then
    echo $0: usage: init.sh local_ip sgx_mem_size, example: init.sh 172.168.10.102 64G
    exit 1
fi

local_ip=$1
sgx_mem_size=$2

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/training/jdk8_spark SPARK_LOCAL_IP=$local_ip SPARK_USER=root G_SGX_SIZE=$sgx_mem_size && \
cd /opt/jdk8/bin && \
ln -s /training/jdk8_spark/java.sig java.sig && \
ln -s /training/jdk8_spark/java.manifest.sgx java.manifest.sgx && \
ln -s /training/jdk8_spark/java.token java.token
