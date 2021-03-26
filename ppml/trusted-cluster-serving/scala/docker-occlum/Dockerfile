# stage.1 Redis
FROM ubuntu:18.04 as redis-tls

WORKDIR /opt

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        git build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/openssl/openssl \ 
        && git checkout tags/OpenSSL_1_1_1 -b OpenSSL_1_1_1
RUN ./config \
        --openssldir=/usr/local/occlum/ssl \
        --with-rand-seed=rdcpu \
        no-zlib no-async no-tests
RUN make -j `getconf _NPROCESSORS_ONLN` && make install

RUN git clone https://github.com/redis/redis.git && git checkout 6.0.6
RUN make -j `getconf _NPROCESSORS_ONLN` BUILD_TLS=yes && make PREFIX=/opt/redis install

# stage. 2 Flink jobmanager
FROM ubuntu:18.04 as jobmanager

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-11-jdk aria2 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG FLINK_VERSION=1.10.1
ENV FLINK_VERSION ${FLINK_VERSION}

# flink
RUN cd /opt && \
    wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar -zxvf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz
RUN ls -al /opt

# stage.3 analytics-zoo
FROM ubuntu:18.04 as analytics-zoo
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG ANALYTICS_ZOO_VERSION=0.10.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.1
ARG SPARK_VERSION=2.4.3
ARG SPARK_MAJOR_VERSION=2.4
ENV ANALYTICS_ZOO_VERSION	${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV JAVA_HOME			/opt/jdk8
ENV PATH			${JAVA_HOME}/bin:${PATH}

COPY --from=flink /opt/jdk  /opt/jdk8

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git maven
RUN git clone https://github.com/intel-analytics/analytics-zoo.git && \
    cd analytics-zoo && \
    export MAVEN_OPTS="-XX:ReservedCodeCacheSize=512m -XX:MaxPermSize=3G \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn clean package -DskipTests -Dspark.version=${SPARK_VERSION} \
        -Dbigdl.artifactId=bigdl-SPARK_$SPARK_MAJOR_VERSION -P spark_2.4+ -Dbigdl.version=${BIGDL_VERSION}
RUN cd /opt && \
    mkdir resnet50 && \
    cd resnet50 && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.bin/download" -O resnet_v1_50.bin && \
    wget -c "https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/openvino/2018_R5/resnet_v1_50.xml/download" -O resnet_v1_50.xml

# stage.4 az ppml occlum
FROM ubuntu:18.04
ARG ANALYTICS_ZOO_VERSION=0.10.0-SNAPSHOT
ARG BIGDL_VERSION=0.12.1
ARG SPARK_VERSION=2.4.3
ARG FLINK_VERSION=1.10.1
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME				/opt/jdk8
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

RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /ppml/trusted-cluster-serving/redis && \
    mkdir -p /ppml/trusted-cluster-serving/redis/work/keys && \
    mkdir -p /ppml/trusted-cluster-serving/redis/work/passowrd && \
    mkdir -p /ppml/trusted-cluster-serving/java/work && \
    mkdir -p /ppml/trusted-cluster-serving/java/work/keys && \
    mkdir -p /ppml/trusted-cluster-serving/java/work/passowrd && \
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
COPY --from=analytics-zoo /analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar /ppml/trusted-cluster-serving/java/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar
COPY --from=analytics-zoo /analytics-zoo/zoo/target/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar /ppml/trusted-cluster-serving/java/work/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-http.jar
COPY --from=analytics-zoo /opt/resnet50/ /ppml/trusted-cluster-serving/java/work/models/resnet50

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev pkg-config && \
    apt-get install -y netcat && \
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
ADD ./java.Makefile /ppml/trusted-cluster-serving/java/Makefile
ADD ./java.manifest.template /ppml/trusted-cluster-serving/java/java.manifest.template
ADD ./init-java.sh /ppml/trusted-cluster-serving/java/init-java.sh
ADD ./start-flink-jobmanager.sh /ppml/trusted-cluster-serving/java/start-flink-jobmanager.sh
ADD ./start-flink-taskmanager.sh /ppml/trusted-cluster-serving/java/start-flink-taskmanager.sh
ADD ./start-http-frontend.sh /ppml/trusted-cluster-serving/java/start-http-frontend.sh
ADD ./cluster-serving-config.yaml /ppml/trusted-cluster-serving/java/config.yaml
ADD ./init-cluster-serving.sh /ppml/trusted-cluster-serving/java/init-cluster-serving.sh
ADD ./start-cluster-serving-job.sh /ppml/trusted-cluster-serving/java/start-cluster-serving-job.sh
ADD ./start-all.sh /ppml/trusted-cluster-serving/start-all.sh

RUN chmod a+x /ppml/trusted-cluster-serving/redis/init-redis.sh && \
    chmod a+x /ppml/trusted-cluster-serving/redis/start-redis.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/init-java.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/start-flink-jobmanager.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/start-flink-taskmanager.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/start-http-frontend.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/init-cluster-serving.sh && \
    chmod a+x /ppml/trusted-cluster-serving/java/start-cluster-serving-job.sh && \
    chmod a+x /ppml/trusted-cluster-serving/start-all.sh
