#!/usr/bin/bash

export MASTER=192.168.0.112
export WORKERS=(192.168.0.112 192.168.0.113)

export TRUSTED_BIGDATA_ML_DOCKER=intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT
export TRUSTED_CLUSTER_SERVING_DOCKER=intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT

export KEYS_PATH=/opt/analytics-zoo-ppml/keys
export SECURE_PASSWORD_PATH=/opt/analytics-zoo-ppml/password
