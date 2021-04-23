export HTTP_PROXY_HOST=child-prc.intel.com
export HTTP_PROXY_PORT=913
export HTTPS_PROXY_HOST=child-prc.intel.com
export HTTPS_PROXY_PORT=913
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
    --build-arg no_proxy=x.x.x.x \
    -t intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT -f ./Dockerfile .
