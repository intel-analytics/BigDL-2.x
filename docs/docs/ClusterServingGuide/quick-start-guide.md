# Quick Start Guide
This guide provides all supported quick start choices of Cluster Serving.

### Docker
Docker quick start is provided in [Programming Guide Quick Start](ProgrammingGuide.md#quick-start)

### Manually Install
In terminal, install Redis, and export `REDIS_HOME`
```
$ export REDIS_VERSION=5.0.5
$ wget http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar xzf redis-${REDIS_VERSION}.tar.gz && \
    rm redis-${REDIS_VERSION}.tar.gz && \
    cd redis-${REDIS_VERSION} && \
    make
$ export REDIS_HOME=$(pwd)
```
install Flink, and export `FLINK_HOME`
```
$ export FLINK_VERSION=1.11.2
$ wget https://archive.apache.org/dist/flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar xzf flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    rm flink-${FLINK_VERSION}-bin-scala_2.11.tgz && 
    cd flink-${FLINK_VERSION}
$ export FLINK_HOME=$(pwd)
```
install Cluster Serving by
```
pip install analytics-zoo-serving
```
then in terminal call
```
$ cluster-serving-init
```
```
$ cluster-serving-start
```

```
img = cv2.imread(path)
img = cv2.resize(img, (224, 224))
data = cv2.imencode(".jpg", img)[1]
img_encoded = base64.b64encode(data).decode("utf-8")
result = input_api.enqueue(key, t={"b64": img_encoded})
```
