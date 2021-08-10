#!/bin/bash

set -x

local_ip=$LOCAL_IP
sgx_mem_size=$SGX_MEM_SIZE

make SGX=1 GRAPHENEDIR=/home/sdp/qiyuan/redis_test/graphene \
	JDK_HOME=/opt/jdk \
	EXECDIR=/bin \
	THIS_DIR=/home/sdp/hangrui/myzoo/analytics-zoo/ppml/trusted-big-data-ml  \
	WORK_DIR=/home/sdp/hangrui/myzoo/analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene/bashexm

