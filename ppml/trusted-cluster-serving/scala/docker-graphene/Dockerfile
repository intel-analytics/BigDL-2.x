# stage.1 graphene
FROM ubuntu:18.04 AS graphene

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y autoconf bison build-essential coreutils gawk git libcurl4-openssl-dev libprotobuf-c-dev protobuf-c-compiler python3-protobuf wget
RUN git clone https://github.com/analytics-zoo/graphene.git /graphene
RUN cd /graphene && \
    git fetch origin branch-0.2 && \
    git checkout branch-0.2
RUN cd /graphene/Pal/src/host/Linux-SGX && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git linux-sgx-driver && \
    cd linux-sgx-driver && \
    git checkout DCAP_1.7 && \
    cp -r driver/linux/* .
RUN cd /graphene && \
    ISGX_DRIVER_PATH=/graphene/Pal/src/host/Linux-SGX/linux-sgx-driver make -s -j4 SGX=1 WERROR=1 && true
RUN for f in $(find /graphene/Runtime -type l); do cp --remove-destination $(realpath $f) $f; done
RUN /graphene/Scripts/download --output redis.tar.gz --sha256 f7ded6c27d48c20bc78e797046c79b6bc411121f0c2d7eead9fea50d6b0b6290 --url https://github.com/antirez/redis/archive/6.0.5.tar.gz

# stage.2 jdk & flink
FROM ubuntu:18.04 as flink
ARG FLINK_VERSION=1.10.1
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV FLINK_VERSION ${FLINK_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz
RUN ls -al /opt

# stage.3 analytics-zoo
FROM ubuntu:18.04 as analytics-zoo
ARG ANALYTICS_ZOO_VERSION=0.10.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.1
ARG SPARK_VERSION=2.4.3
ENV ANALYTICS_ZOO_VERSION	${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV ANALYTICS_ZOO_HOME		/analytics-zoo-${ANALYTICS_ZOO_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip 
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/docker/zoo/download-analytics-zoo.sh && \
    chmod a+x ./download-analytics-zoo.sh
RUN ./download-analytics-zoo.sh

# stage.4 az ppml
FROM ubuntu:18.04
ARG ANALYTICS_ZOO_VERSION=0.10.0-SNAPSHOT
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION       ${ANALYTICS_ZOO_VERSION}
ENV FLINK_VERSION		${FLINK_VERSION}
ENV FLINK_HOME			/ppml/trusted-cluster-serving/java/work/flink-${FLINK_VERSION}
ENV ANALYTICS_ZOO_HOME		/ppml/trusted-cluster-serving/java/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME			/opt/jdk8
ENV PATH			${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP			127.0.0.1
ENV SGX_MEM_SIZE		64G
ENV REDIS_PORT			6379

RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /ppml/trusted-cluster-serving/redis && \
    mkdir -p /ppml/trusted-cluster-serving/redis/work/keys && \
    mkdir -p /ppml/trusted-cluster-serving/java/work && \
    mkdir -p /ppml/trusted-cluster-serving/java/work/keys && \
    mkdir -p /ppml/trusted-cluster-serving/java/work/data && \
    mkdir -p /ppml/trusted-cluster-serving/java/work/models

COPY --from=graphene /graphene/Scripts /graphene/Scripts
COPY --from=graphene /graphene/Runtime/ /graphene/Runtime
COPY --from=graphene /graphene/python /graphene/python
COPY --from=graphene /graphene/Pal /graphene/Pal
COPY --from=graphene /graphene/Pal/src/host/Linux-SGX/generated_offsets.py /graphene/python/
COPY --from=graphene /graphene/Examples/common_tools/get_deps.sh /graphene/get_deps.sh
COPY --from=graphene /graphene/Tools/argv_serializer /graphene/Tools
COPY --from=graphene /redis.tar.gz /ppml/trusted-cluster-serving/redis
COPY --from=flink /opt/jdk  /opt/jdk8
COPY --from=flink /opt/flink-${FLINK_VERSION} /ppml/trusted-cluster-serving/java/work/flink-${FLINK_VERSION}
COPY --from=analytics-zoo /analytics-zoo-${ANALYTICS_ZOO_VERSION} /ppml/trusted-cluster-serving/java/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev && \
    mkdir -p /graphene/Pal/src/host/Linux-SGX/signer && \
    cd /graphene/Pal/src/host/Linux-SGX/signer && \
    openssl genrsa -3 -out enclave-key.pem 3072

RUN mkdir -p /ppml/trusted-cluster-serving/redis/src && \
    tar -C /ppml/trusted-cluster-serving/redis/src --strip-components=1 -xf /ppml/trusted-cluster-serving/redis/redis.tar.gz && \
    make BUILD_TLS=yes -C /ppml/trusted-cluster-serving/redis/src && \
    cp /ppml/trusted-cluster-serving/redis/src/src/redis-server /ppml/trusted-cluster-serving/redis/redis-server

ADD ./redis.Makefile /ppml/trusted-cluster-serving/redis/Makefile
ADD ./redis-server.manifest.template /ppml/trusted-cluster-serving/redis/redis-server.manifest.template
ADD ./init-redis.sh /ppml/trusted-cluster-serving/redis/init-redis.sh
ADD ./start-redis.sh /ppml/trusted-cluster-serving/redis/start-redis.sh

RUN chmod a+x /ppml/trusted-cluster-serving/redis/init-redis.sh && \
    chmod a+x /ppml/trusted-cluster-serving/redis/start-redis.sh
