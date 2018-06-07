#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM ubuntu:18.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG ANALYTICS_ZOO_VERSION=0.1.0-SNAPSHOT
ARG SPARK_VERSION=2.2.0
ENV ANALYTICS_ZOO_VERSION_ENV   ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION_ENV           ${SPARK_VERSION}
ENV SPARK_HOME                  /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME          /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                   /opt/jdk
ENV CONDA_HOME                  /opt/work/conda
ENV PATH                        ${JAVA_HOME}/bin:${CONDA_HOME}/bin:${PATH}
ENV TENSOR_FLOW_VERION          1.2.0
ENV KERAS_1_VERSION             1.2.2
ENV KERAS_2_VERSION             2.1.6

RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip maven git
#java
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk
#python-conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.3.31-Linux-x86_64.sh -f -b -p ${CONDA_HOME} && \
    rm Miniconda3-4.3.31-Linux-x86_64.sh && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda update --all --quiet --yes && \
    conda clean -tipsy && \
    conda create -y -n py27 python=2.7 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytes                                      t pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_1_VERSION && \
    conda create -y -n py35 python=3.5 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytes                                      t pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_1_VERSION && \
    conda create -y -n py36 python=3.6 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytes                                      t pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_1_VERSION && \
    conda info --env
#spark
RUN wget https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
