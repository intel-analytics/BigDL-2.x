FROM krallin/ubuntu-tini AS tini
FROM occlum/occlum:0.24.0-ubuntu18.04 AS occlum

ENV SPARK_HOME /opt/spark

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

COPY --from=tini /usr/local/bin/tini /sbin/tini

# stage.2 jdk & scala & spark & hadoop
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG SPARK_VERSION=3.1.2
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url 
ENV SPARK_VERSION	${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk$JDK_VERSION
ENV PATH                ${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip patch zip git maven
ADD hadoop-common-shell.patch /opt/hadoop-common-shell.patch

#java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

#scala
RUN cd / && wget -c https://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.tgz && \
    (cd / && gunzip < scala-2.11.8.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-2.11.8.tgz

#maven
RUN cd /opt && \
    wget https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && \
    tar -zxvf apache-maven-3.6.3-bin.tar.gz

# Download & prepare Spark 3.1.2
WORKDIR /opt
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz -P /opt/ && \
    tar -zxvf /opt/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop3.2 /opt/spark && \
    rm /opt/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp -r /opt/spark/examples /bin/examples && \
    cp -r /opt/spark/kubernetes/tests /opt/spark/tests
    cp /opt/spark/conf/log4j.properties.template /opt/spark/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> /opt/spark/conf/log4j.properties
    
# Remove fork with libhadoop.so and spark-network-common_2.12-3.1.2.jar
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/ && \
    rm -f /opt/spark/jars/spark-network-common_2.12-3.0.0.jar && \
    wget https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar -P /opt/spark/jars

    wget -O /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar && \
    mv /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar /opt/spark/jars/spark-${SPARK_VERSION}/jars

# spark modification
# org.apache.spark.util.Utils to disable chmod fork
RUN cd /opt && \
    git clone https://github.com/apache/spark.git && \
    cd spark && \
    git checkout tags/v3.1.2 -b branch-3.1.2 && \
    git apply /opt/spark-3.1.2.patch && \
    git status && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean install -pl core && \
    cd resource-managers/kubernetes/core && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package && \
    ls /opt/spark/core/target/spark-core_2.12-$SPARK_VERSION.jar && \
    ls /opt/spark/resource-managers/kubernetes/core/target/spark-kubernetes_2.12-3.1.2.jar

COPY ./entrypoint.sh /opt/
COPY ./init.sh /opt/

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/init.sh

WORKDIR /opt/

ENTRYPOINT [ "/opt/entrypoint.sh" ]
