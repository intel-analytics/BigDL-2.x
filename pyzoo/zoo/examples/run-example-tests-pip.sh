#!/usr/bin/env bash

echo "#1 start example test for textclassification"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then
    echo "analytics-zoo-data/data/20news-18828.tar.gz already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/20news-18828.tar.gz -C analytics-zoo-data/data/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 2 \
    --data_path analytics-zoo-data/data/20news-18828 \
    --embedding_path analytics-zoo-data/data/glove.6B
