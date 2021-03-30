# stage.1 Redis
FROM ubuntu:18.04 as redis-tls

WORKDIR /opt

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git build-essential coreutils \
        && \
    apt-get clean

RUN apt-get install -y ca-certificates

RUN git clone https://github.com/openssl/openssl.git 

RUN cd openssl && \
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

# stage. 2 Flink jobmanager
FROM ubuntu:18.04 as flink-jobmanager

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk wget \
        && \
    apt-get clean

ARG FLINK_VERSION=1.10.1
ENV FLINK_VERSION ${FLINK_VERSION}

# flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz
RUN ls -al /opt

# stage. 3 Flink taskmanager
FROM FROM occlum/occlum:0.21.0-ubuntu18.04 as flink-taskmanager-occlum
ARG FLINK_VERSION=1.10.1
ENV FLINK_VERSION ${FLINK_VERSION}

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk wget \
        && \
    apt-get clean

# copy flink
COPY --from=flink-jobmanager /opt/flink-${FLINK_VERSION} /opt/

ADD ./build_occlum_taskmanager.sh /opt/build_occlum_taskmanager.sh
ADD ./run_occlum_taskmanager.sh /opt/run_occlum_taskmanager.sh

# stage.3 analytics-zoo

# stage.4 az ppml occlum
FROM ubuntu:18.04
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
ENV FLINK_HOME				/ppml/trusted-cluster-serving/java/work/flink-${FLINK_VERSION}
ENV FLINK_JOB_MANAGER_IP		127.0.0.1
ENV FLINK_JOB_MANAGER_REST_PORT		8081
ENV FLINK_JOB_MANAGER_RPC_PORT		6123
ENV FLINK_TASK_MANAGER_IP		127.0.0.1
ENV FLINK_TASK_MANAGER_DATA_PORT	6124
ENV FLINK_TASK_MANAGER_RPC_PORT		6125
ENV FLINK_TASK_MANAGER_TASKSLOTS_NUM	1
ENV CORE_NUM                            2

COPY --from=analytics-zoo /opt/resnet50/ /ppml/trusted-cluster-serving/java/work/models/resnet50

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev pkg-config && \
    apt-get install -y netcat

