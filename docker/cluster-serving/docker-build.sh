docker build \
    --build-arg http_proxy=http://10.239.4.101:913/ \
    --build-arg https_proxy=https://10.239.4.101:913/ \
    --rm -t analytics-zoo/cluster-serving:0.7.0-spark_2.4.3 \
    -f ./DockerFile .
