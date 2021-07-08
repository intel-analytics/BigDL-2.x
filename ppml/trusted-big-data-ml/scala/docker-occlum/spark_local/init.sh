export http_proxy=your_proxy:proxy_port
export https_proxy=your_proxy:proxy_port
apt-get update
apt-get install -y openjdk-11-jdk

mkdir -p /bin/examples/jars
wget -P /bin/examples/jars/spark-network-common_2.11-2.4.3.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.11-2.4.3.jar
