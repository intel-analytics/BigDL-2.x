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
ADD ./init-occlum-taskmanager.sh /opt/init-occlum-taskmanager.sh
ADD ./start-redis.sh /opt/start-redis.sh
ADD ./start-flink-jobmanager.sh /opt/start-flink-jobmanager.sh
ADD ./start-flink-taskmanager.sh /opt/start-flink-taskmanager.sh
ADD ./start-http-frontend.sh /opt/start-http-frontend.sh
ADD ./start-local-cluster-serving.sh /opt/start-local-cluster-serving.sh
ADD ./start-all.sh /opt/start-all.sh
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

RUN cd /opt && \
    mkdir resnet50 && \
    cd resnet50 && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin/download" -O resnet_v1_50.bin && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml/download" -O resnet_v1_50.xml

RUN mkdir -p /opt/analytics-zoo && \
    cd analytics-zoo && \
    wget -c "https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/zoo/analytics-zoo-bigdl_0.12.1-spark_2.4.3/0.10.0-SNAPSHOT/analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.10.0-20210322.210553-65-cluster-serving-all.zip" && \
    tar -zxvf analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.10.0-20210322.210553-65-cluster-serving-all.zip && \
    rm analytics-zoo-bigdl_0.12.1-spark_2.4.3-0.10.0-20210322.210553-65-cluster-serving-all.zip


