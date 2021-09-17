#!/bin/bash

${SPARK_HOME}/bin/spark-submit \
    --master k8s://https://192.168.0.113:6443 \
    --deploy-mode cluster \
    --name spark-pi \
    --class org.apache.spark.examples.SparkPi \
    --conf spark.executor.instances=1 \
    --conf spark.rpc.netty.dispatcher.numThreads=32 \
    --conf spark.kubernetes.container.image=intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-occlum:0.12-SNAPSHOT \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.driver.podTemplateFile=./executor.yaml \
    --conf spark.kubernetes.executor.podTemplateFile=./executor.yaml \
    local:/bin/examples/jars/spark-examples_2.12-3.0.0.jar
