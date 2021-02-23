# JAVA SPARK Graphene
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.


## How To Build

```bash
export http_proxy=http://proxy-chain.intel.com:911
export https_proxy=http://proxy-chain.intel.com:912
export HTTP_PROXY_HOST=proxy-chain.intel.com
export HTTP_PROXY_PORT=911
export HTTPS_PROXY_HOST=proxy-chain.intel.com
export HTTPS_PROXY_PORT=912
sudo docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg HTTP_PROXY_HOST=$HTTP_PROXY_HOST \
    --build-arg HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
    --build-arg HTTPS_PROXY_HOST=$HTTPS_PROXY_HOST \
    --build-arg HTTPS_PROXY_PORT=$HTTPS_PROXY_PORT \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=http://10.239.45.10:8081/repository/raw/jdk/jdk-8u192-linux-x64.tar.gz \
    --build-arg no_proxy=10.239.45.10 \
    -t analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest -f ./Dockerfile .
sudo docker tag analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest 10.239.47.32/arda/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest
```

