--------
# Docker images and builders for Analytics-Zoo

## Analytics-Zoo in Docker

### By default, the Analytics-Zoo image has installed below packages:
* git
* maven
* Oracle jdk 1.8.0_152 (in /opt/jdk1.8.0_152)
* python 2.7.6
* pip
* numpy
* scipy
* pandas
* scikit-learn
* matplotlib
* seaborn
* jupyter
* wordcloud
* moviepy
* opencv
* spark-${SPARK_VERSION} (in /opt/work/spark-${SPARK_VERSION})
* Analytics-Zoo distribution (in /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION})
* Analytics-Zoo source code (in /opt/work/analytics-zoo)

### The work dir for Analytics-Zoo is /opt/work.

* download-analytics-zoo.sh is used for downloading Analytics-Zoo distributions.
* start-notebook.sh is used for starting the jupyter notebook. You can specify the environment settings and spark settings to start a specified jupyter notebook.
* analytics-Zoo-${ANALYTICS_ZOO_VERSION} is the Analytics-Zoo home of Analytics-Zoo distribution.
* analytics-zoo-SPARK_x.x-x.x.x-dist.zip is the zip file of Analytics-Zoo distribution.
* spark-${SPARK_VERSION} is the Spark home.
* analytics-zoo is cloned from https://github.com/intel-analytics/analytics-zoo, contains apps, examples using analytics-zoo.

## How to build it.

### By default, you can build a Analytics-Zoo:default image with latest nightly-build Analytics-Zoo distributions:

    sudo docker build --rm -t intelanalytics/analytics-zoo:default .

### If you need http and https proxy to build the image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --rm -t intelanalytics/analytics-zoo:default .

### You can also specify the ANALYTICS_ZOO_VERSION and SPARK_VERSION to build a specific Analytics-Zoo image:

    sudo docker build \
        --build-arg http_proxy=http://your-proxy-host:your-proxy-port \
        --build-arg https_proxy=https://your-proxy-host:your-proxy-port \
        --build-arg ANALYTICS_ZOO_VERSION=0.1.0 \
        --build-arg SPARK_VERSION=2.2.0 \
        --rm -t intelanalytics/analytics-zoo:0.1.0-spark-2.2.0 .

## How to use the image.

### To start a notebook directly with a specified port(e.g. 12345). You can view the notebook on http://[host-ip]:12345

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token intelanalytics/analytics-zoo:default

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token intelanalytics/analytics-zoo:default

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token intelanalytics/analytics-zoo:0.1.0-spark-2.2.0

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token intelanalytics/analytics-zoo:0.1.0-spark-2.2.0

### If you need http and https proxy in your environment:

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port intelanalytics/analytics-zoo:default

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port  intelanalytics/analytics-zoo:default

    sudo docker run -it --rm -p 12345:12345 -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port  intelanalytics/analytics-zoo:0.1.0-spark-2.2.0

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token -e http_proxy=http://your-proxy-host:your-proxy-port -e https_proxy=https://your-proxy-host:your-proxy-port intelanalytics/analytics-zoo:0.1.0-spark-2.2.0

### You can also start the container first

    sudo docker run -it --rm --net=host -e NotebookPort=12345 -e NotebookToken=your-token intelanalytics/analytics-zoo:default bash

### In the container, after setting proxy and ports, you can start the Notebook by:

    /opt/work/start-notebook.sh

