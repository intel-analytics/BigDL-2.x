ARG JOBMANAGER_MEMORY_PROCESS_SIZE=10g
ARG TASKMANAGER_MEMORY_PROCESS_SIZE=10g
ARG TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE=2097152b
ARG TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE=2048mb
ARG RUNTIME_K8S_FLINK_NAME_SPACE=stream
ARG RUNTIME_K8S_FLINK_SERVICE_ACCOUNT=stream
ARG RUNTIME_K8S_FLINK_IMAGE=${RUNTIME_K8S_FLINK_IMAGE}
ARG RUNTIME_K8S_FLINK_CLUSER_ID=cluster-serving
ARG FLINK_UI_HOST=localhost
ARG FLINK_UI_PORT=8081
ARG REDIS_HOME=/opt/redis-5.0.5
ARG REDIS_STORAGE=/opt/redis-storage
ARG REDIS_PORT=36379
ARG REDIS_VERSION=5.0.5

#stage.1 redis
FROM ubuntu:18.04 as redis
ARG REDIS_VERSION

RUN apt-get update --fix-missing && \
    apt-get install -y wget make gcc && \
    wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-*.tar.gz -C /opt && \
    rm redis-*.tar.gz && \
    cd /opt/redis-${REDIS_VERSION} && \
    make && \
    echo "bind 0.0.0.0" >> /opt/redis-${REDIS_VERSION}/redis.conf

#stage.2 flink
FROM flink:1.13.0-scala_2.11
ARG JOBMANAGER_MEMORY_PROCESS_SIZE
ARG TASKMANAGER_MEMORY_PROCESS_SIZE
ARG TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE
ARG TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE
ARG RUNTIME_K8S_FLINK_NAME_SPACE
ARG RUNTIME_K8S_FLINK_SERVICE_ACCOUNT
ARG RUNTIME_K8S_FLINK_IMAGE
ARG RUNTIME_K8S_FLINK_CLUSER_ID
ARG FLINK_UI_HOST
ARG FLINK_UI_PORT
ARG REDIS_HOME
ARG REDIS_STORAGE
ARG REDIS_PORT

ENV JOBMANAGER_MEMORY_PROCESS_SIZE              ${JOBMANAGER_MEMORY_PROCESS_SIZE}
ENV TASKMANAGER_MEMORY_PROCESS_SIZE             ${TASKMANAGER_MEMORY_PROCESS_SIZE}
ENV TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE       ${TASKMANAGER_MEMORY_TASK_OFF_HEAP_SIZE}
ENV TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE  ${TASKMANAGER_MEMORY_FRAMEWORK_OFF_HEAP_SIZE}
ENV RUNTIME_K8S_FLINK_NAME_SPACE                ${RUNTIME_K8S_FLINK_NAME_SPACE}
ENV RUNTIME_K8S_FLINK_SERVICE_ACCOUNT           ${RUNTIME_K8S_FLINK_SERVICE_ACCOUNT}
ENV RUNTIME_K8S_FLINK_IMAGE                     ${RUNTIME_K8S_FLINK_IMAGE}
ENV RUNTIME_K8S_FLINK_CLUSER_ID                 ${RUNTIME_K8S_FLINK_CLUSER_ID}
ENV FLINK_UI_HOST                               ${FLINK_UI_HOST}
ENV FLINK_UI_PORT                               ${FLINK_UI_PORT}
ENV REDIS_HOME                                  ${REDIS_HOME}
ENV REDIS_PORT                                  ${REDIS_PORT}
ENV CONFIG_DIR                                  /opt/work/conf
ENV REDIS_STORAGE                               ${REDIS_STORAGE}

COPY --from=redis ${REDIS_HOME} ${REDIS_HOME}
COPY ./download-cluster-serving-all-zip.sh /hyper-zoo/
COPY ./*.yaml ${CONFIG_DIR}/
COPY ./start*.sh /opt/work/scripts/
COPY ./models /opt/models

RUN apt-get update --fix-missing && \
    apt-get -y install vim && \
    mkdir ${REDIS_STORAGE} && \
    cd /hyper-zoo && \
    chmod a+x /hyper-zoo/download-cluster-serving-all-zip.sh && \
    /hyper-zoo/download-cluster-serving-all-zip.sh && \
    cp /hyper-zoo/cluster-serving/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.11.0-SNAPSHOT-*jar /opt/work/ && \
    cd .. && \
    rm -rf /hyper-zoo && \
    chmod a+x /opt/work/scripts/*
