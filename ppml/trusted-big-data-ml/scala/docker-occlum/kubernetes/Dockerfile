
FROM krallin/ubuntu-tini AS tini
FROM occlum/occlum:0.23.7-ubuntu18.04 AS occlum

ENV SPARK_HOME /opt/spark

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 


WORKDIR /root
RUN git clone https://github.com/occlum/occlum.git && \
    cd occlum && \
    source /opt/intel/sgxsdk/environment && \
    git checkout -b 0.23.7 0.23.7 && \
    git apply 
    make submodule && \
    OCCLUM_RELEASE_BUILD=1 make && \
    make install && \
    cp -r demos /root/demos && \
    rm -rf /root/occlum

COPY --from=tini /usr/local/bin/tini /sbin/tini

WORKDIR /opt
RUN wget https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz -P /opt/ && \
    tar -xvzf /opt/spark-3.0.0-bin-hadoop2.7.tgz && mv /opt/spark-3.0.0-bin-hadoop2.7 /opt/spark && \
    rm /opt/spark-3.0.0-bin-hadoop2.7.tgz && \
    cp -r /opt/spark/examples /bin/examples && \
    cp -r /opt/spark/kubernetes/tests /opt/spark/tests

RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/ && \
    wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/spark-network-common_2.12-3.0.0.jar -P /opt/spark/jars

COPY ./entrypoint.sh /opt/
COPY init.sh /opt/

WORKDIR /opt/


ENTRYPOINT [ "/opt/entrypoint.sh" ]
