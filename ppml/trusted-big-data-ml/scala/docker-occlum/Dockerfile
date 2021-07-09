FROM occlum/occlum:0.23.0-ubuntu18.04 AS occlum
ENV LOCAL_IP your_local_ip
ENV BIGDL_VERSION bigl_version
ENV SPARK_VERSION spark_version
ENV ANALYTICS_ZOO_VERSION long_az_version
ENV http_proxy $http_proxy
ENV https_proxy $https_proxy
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk
RUN mkdir -p /bin/examples/jars
RUN mkdir -p /ppml/docker-occlum && cd /ppml/docker-occlum && \
    wget -P /ppml/docker-occlum https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz && \
    tar -xvzf /ppml/docker-occlum/spark-2.4.3-bin-hadoop2.7.tgz  &&  \
    wget -P /bin/examples/jars/spark-network-common_2.11-2.4.3.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.11-2.4.3.jar && \
    ls /ppml/docker-occlum
ADD ./hosts /ppml/docker-occlum/hosts
ADD ./README.md  /ppml/docker-occlum/README.md 
ADD ./run_spark_on_occlum_glibc.sh /ppml/docker-occlum/run_spark_on_occlum_glibc.sh
ADD ./start-spark-local.sh /ppml/docker-occlum/start-spark-local.sh
ADD ./bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar /ppml/docker-occlum/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar
ADD ./analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar /ppml/docker-occlum/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar
