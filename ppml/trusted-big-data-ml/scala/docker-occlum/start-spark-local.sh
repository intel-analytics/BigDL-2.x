sudo docker run -itd \
	--net=host \
	--name=occlum-spark-local \
	--cpuset-cpus 10-14 \
	--device=/dev/sgx \
	-v ../docker-occlum/:/ppml/docker-occlum \
	-e LOCAL_IP=$LOCAL_IP \
	occlum/occlum:0.23.0-ubuntu18.04 \
	bash /ppml/docker-occlum/init.sh && /ppml/docker-occlum/run_spark_on_occlum_glibc.sh $1 && tail -f /dev/null
