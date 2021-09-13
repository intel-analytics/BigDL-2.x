#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 GRAPHENEDIR=/graphene THIS_DIR=/ppml/trusted-big-data-ml  SPARK_LOCAL_IP=$local_ip SPARK_USER=root G_SGX_SIZE=$sgx_mem_size

cd /dev
mkdir sgx
ln -s /dev/sgx_enclave /dev/sgx/enclave
ln -s /dev/sgx_provision /dev/sgx/provision

cd /ppml/trusted-big-data-ml
