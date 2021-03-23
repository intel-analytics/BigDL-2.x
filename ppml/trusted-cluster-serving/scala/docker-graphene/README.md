# trusted-cluster-serving
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.

## How To Build
Set parameter before you build: <br>
```bash
export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export JDK_URL=http://your-http-url-to-download-jdk
```
Then build docker image: <br>
```bash
./build-docker-image.sh
```

## How To Run
### Prepare the keys
The ppml in analytics zoo need secured keys to enable flink TLS, https and tlse enabled Redis, you need to prepare the secure keys and keystores. <br>
```bash
./generate-keys.sh
```
You also need to store the password you used in previous step in a secured file: <br>
```bash
./generate-password.sh
```

### Run the PPML Docker image
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
Set the parameter of path: <br>
```bash
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export SECURE_PASSWORD_PATH=the_dir_path_of_your_prepared_password
export LOCAL_IP=your_local_ip_of_the_sgx_server
```
Then run the example with docker: <br>
```bash
./run_docker_local_example.sh
```

#### In distributed mode
##### setup passwordless ssh login to all the nodes.
##### config the environments for master, workers, docker image and security keys/passowrd files.
```bash
nano environments.sh
```
##### start the distributed cluster serving
```bash
./start-distributed-cluster-serving.sh
```
##### stop the distributed cluster serving 
```bash
./stop-distributed-cluster-serving.sh
```
