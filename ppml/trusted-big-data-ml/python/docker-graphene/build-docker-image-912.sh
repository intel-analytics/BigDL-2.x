export HTTP_PROXY_HOST=proxy-chain.intel.com
export HTTP_PROXY_PORT=912
export HTTPS_PROXY_HOST=proxy-chain.intel.com
export HTTPS_PROXY_PORT=912
export JDK_URL=http://10.239.45.10:8081/repository/raw/jdk/jdk-8u192-linux-x64.tar.gz

#export tag=0.12
export tag=0.12-k8s
#export tag=0.12-SNAPSHOT-1.2-rc1


sudo docker build \
    --build-arg http_proxy=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT \
    --build-arg https_proxy=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT \
    --build-arg HTTP_PROXY_HOST=$HTTP_PROXY_HOST \
    --build-arg HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
    --build-arg HTTPS_PROXY_HOST=$HTTPS_PROXY_HOST \
    --build-arg HTTPS_PROXY_PORT=$HTTPS_PROXY_PORT \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=$JDK_URL \
    --build-arg no_proxy=10.239.45.10 \
    -t intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-python-gramine:$tag -f ./Dockerfile .

sudo docker tag intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-python-gramine:$tag 10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-python-gramine:$tag
sudo docker push 10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-python-gramine:$tag
