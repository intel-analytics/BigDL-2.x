export HTTP_PROXY_HOST=proxy-chain.intel.com
export HTTP_PROXY_PORT=911
export HTTPS_PROXY_HOST=proxy-chain.intel.com
export HTTPS_PROXY_PORT=912
export JDK_URL=http://10.239.45.10:8081/repository/raw/jdk/jdk-8u192-linux-x64.tar.gz

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
    -t intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene-1.2-rc1:0.12.0-SNAPSHOT -f ./Dockerfile .

sudo docker tag intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene-1.2-rc1:0.12.0-SNAPSHOT 10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene-1.2-rc1:0.12.0-SNAPSHOT
sudo docker push 10.239.45.10/arda/intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene-1.2-rc1:0.12.0-SNAPSHOT
