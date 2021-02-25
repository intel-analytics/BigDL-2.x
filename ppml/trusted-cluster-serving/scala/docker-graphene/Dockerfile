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

# stage.2 jdk & flink
FROM ubuntu:18.04 as flink
ARG FLINK_VERSION=1.10.1
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV FLINK_VERSION $FLINK_VERSION

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
