# stage.1 Redis
FROM occlum/occlum:0.21.0-ubuntu18.04 as redis-tls

WORKDIR /opt

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git build-essential coreutils ca-certificates && \
    apt-get clean

RUN git clone https://github.com/openssl/openssl.git && \
    cd openssl && \
    git checkout tags/OpenSSL_1_1_1 -b OpenSSL_1_1_1 && \
    ./config \
    --openssldir=/opt/ssl \
    --with-rand-seed=rdcpu \
    no-zlib no-async no-tests && \
    make -j `getconf _NPROCESSORS_ONLN` && make install

RUN git clone https://github.com/redis/redis.git && \
    cd redis && \
    git checkout 6.0.6 && \
    make -j `getconf _NPROCESSORS_ONLN` BUILD_TLS=yes && make PREFIX=/opt/redis install

# stage. 2 Flink
FROM occlum/occlum:0.21.0-ubuntu18.04 as flink-jobmanager

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk wget && \
    apt-get clean

ARG FLINK_VERSION=1.10.1
ENV FLINK_VERSION ${FLINK_VERSION}

# Download flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz

# stage.3 analytics-zoo
ADD ./build-occlum-taskmanager.sh /opt/build-occlum-taskmanager.sh
ADD ./start-occlum-taskmanager.sh /opt/start-occlum-taskmanager.sh
ADD ./start-jobmanager.sh /opt/start-occlum-taskmanager.sh
ADD ./start-local-cluster-serving.sh /opt/start-local-cluster-serving.sh
ADD ./start-distributed-cluster-serving.sh /opt/start-distributed-cluster-serving.sh

# stage.4 az ppml occlum
FROM occlum/occlum:0.21.0-ubuntu18.04
ARG ANALYTICS_ZOO_VERSION=0.10.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.1
ARG SPARK_VERSION=2.4.3
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME				/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH				${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP				127.0.0.1
ENV SGX_MEM_SIZE			64G
ENV REDIS_PORT				6379
ENV FLINK_VERSION			${FLINK_VERSION}
ENV FLINK_HOME				/opt/flink-${FLINK_VERSION}
ENV FLINK_JOB_MANAGER_IP		127.0.0.1
ENV FLINK_JOB_MANAGER_REST_PORT		8081
ENV FLINK_JOB_MANAGER_RPC_PORT		6123
ENV FLINK_TASK_MANAGER_IP		127.0.0.1
ENV FLINK_TASK_MANAGER_DATA_PORT	6124
ENV FLINK_TASK_MANAGER_RPC_PORT		6125
ENV FLINK_TASK_MANAGER_TASKSLOTS_NUM	1
ENV CORE_NUM                            2

RUN mkdir -p /opt/analytics-zoo && \


RUN wget 

