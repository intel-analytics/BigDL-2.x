export LOCAL_IP=your_local_ip
sudo docker run -itd \
	--net=host \
	--name=occlum-spark-local \
	--cpuset-cpus 10-14 \
	--device=/dev/sgx \
	-v ../../java-spark-occlum/:/java-spark-occlum \
	-e LOCAL_IP=$LOCAL_IP \
	occlum/occlum:0.23.0-ubuntu18.04 \
	bash /java-spark-occlum/spark_local/init.sh && /java-spark-occlum/spark_local/run_spark_on_occlum_glibc.sh $1 && tail -f /dev/null
#docker cp ../../java-spark-occlum/ occlum-spark-local:java-spark-occlum/
