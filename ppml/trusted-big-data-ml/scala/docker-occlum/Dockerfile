FROM occlum/occlum:0.23.0-ubuntu18.04 AS occlum
ENV LOCAL_IP your_local_ip
ENV BIGDL_VERSION bigdl_version
ENV SPARK_VERSION spark_version
ENV ANALYTICS_ZOO_VERSION az_version
ENV http_proxy $http_proxy
ENV https_proxy $https_proxy
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk
RUN mkdir -p /bin/examples/jars
RUN wget -P /bin/examples/jars/spark-network-common_2.11-2.4.3.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.11-2.4.3.jar

