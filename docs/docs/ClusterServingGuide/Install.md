# Install and Run
Currently Analytics Zoo Cluster Serving supports install and run with docker.
## Install and Run with Docker
To install with docker, please see following steps.

1. Install [Docker]()

2. Get Analytics Zoo Cluster Serving docker image.

3. Clone Analytics Zoo repository or download the zip and unzip it.

4. Go to `analytics-zoo/docker/cluster-serving`. Set configuration in `config.yaml`, for details of configuration, see [Configuration Guide]().

5. run `docker run --name cluster-serving --net=host -v $(pwd)/model:/opt/work/model -v $(pwd)/config.yaml:/opt/work/config.yaml analytics-zoo/cluster-serving:0.7.0-spark_2.4.0`
