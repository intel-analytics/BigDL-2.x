# Hyper-zoo serving

* [Overview](#Overview)
* [Build hyper-zoo-serving image](#Buildhyper-zoo-servingimage)
* [Start the hyper-zoo serving container](#Startthehyper-zooservingcontainer)
* [Start services](#Startservices)  
* [Configuration file](#Configurationfile)  
	* [http-conf.yaml](#http-conf.yaml)  
	* [serving-conf.yaml](#serving-conf.yaml) 
* [Use wrk to test service performance](#Usewrktotestserviceperformance)  

##  1. <a name='Overview'></a>Overview
hyper-zoo serving is the way to start cluster-serving in client mode. In the client mode, the required environment and the script(including redis, frontend and serving services) to start the service are packaged in the image. 

* `redis` runs on the client side
* `frontend(http)` runs on the client side
* `serving` runs on the cluster

**First refer to the <a href="#Configurationfile">Configuration file</a>, and then build the image after completing the configuration**

##  2. <a name='Buildhyper-zoo-servingimage'></a>Build hyper-zoo-serving image
The first step is to build a hyper-zoo serving image from the `Dockerfile` in the current directory.   

Use `--build-arg` to set the `env` parameter required when building the image, `<IMAGE_TAG>` is the name of the custom image tag.

The following is the build command：

```shell
docker build \
  --build-arg http_proxy=$HTTP_RROXY \
  --build-arg https_proxy=$HTTPS_RROXY \
  --build-arg no_proxy=$NO_PROXY \
  --rm -t <IMAGE_TAG> -f Dockerfile .
```

**After the image is successfully built, pull the image to each node of the cluster to ensure that the image exists on each node in the cluster**

##  3. <a name='Startthehyper-zooservingcontainer'></a>Start the hyper-zoo serving container

The following is the command to start the container:  

`<CONTAINER_NAME>` is the custom container name, `$KUBER_CONFIG` is the k8s configuration file that needs to be mounted, and `$IMAGE_TAG` is the image tag name of the image built in the previous step.

```shell
docker run -itd
    --net=host \
    --name=<CONTAINER_NAME> \
    -v $KUBER_CONFIG:/root/.kube/config \
    -e RUNTIME_K8S_FLINK_IMAGE=$IMAGE_TAG \
    $IMAGE_TAG bash
```

##  4. <a name='Startservices'></a>Start services
In the `/opt/work/scripts` directory, there is a script file named `start-cluster-serving-all.sh`. This script is responsible for starting the three services of `redis`, `frontend(http)` and `serving`. If you want to start separately, you can copy the startup command in the script and start it separately in the container.

execute script to start services:
```shell
cd /opt/work/scripts
./start-cluster-serving-all.sh
```

##  5. <a name='Configurationfile'></a>Configuration file
###  5.1. <a name='http-conf.yaml'></a>http-conf.yaml
The following is a description of the configuration file parameters:
* modelName: Model name
* modelVersion: Model version
* redisHost: The host where the redis service is running
* redisPort: Redis port
* redisInputQueue:
* redisOutputQueue: 

###  5.2. <a name='serving-conf.yaml'></a>serving-conf.yaml
The following is a description of the configuration file parameters:
* modelPath: Model path
* jobName: This value specifies a certain serving `redisInputQueue` parameter in http-conf.yaml
* modelParallelism: Parallelism of taskmanager
* flinkRestUrl: Flink web UI url
* redisUrl: Reids url, example: `localhost:36379`



##  6. <a name='Usewrktotestserviceperformance'></a>Use wrk to test service performance
On the nodes that can communicate with the cluster, submit a request to the service through the wrk tool to test the performance of the service.

Example of executing wrk command: 
```shell
./wrk/wrk -t4 -c4 -d30s -T30s http://<ip:port>/models/example/versions/1.0/predict --latency -s ./exmaple.lua
```

