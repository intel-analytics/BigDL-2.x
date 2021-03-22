# trusted-cluster-serving
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.

## How To Build
```bash
./build_docker_image.sh your_http_proxy_host your_http_proxy_port your_https_proxy_host your_https_proxy_port http://your-http-url-to-download-jdk
```
For example: <br>
`./build_docker_image.sh 8081 8082 8083 8084 https://www.oracle.com/cn/java/technologies/javase/javase-jdk8-downloads.html`

## How To Run
### Prepare the keys
The ppml in analytics zoo need secured keys to enable flink TLS, https and tlse enabled Redis, you need to prepare the secure keys and keystores. <br>
```bash
./run_docker_create_keys.sh
```
You also need to store the password you used in previous step in a secured file: <br>
```bash
./run_docker_store_pwd.sh
```

### Run the PPML Docker image
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
```bash
./run_docker_local_example.sh the_dir_path_of_your_prepared_keys the_dir_path_of_your_prepared_password your_local_ip_of_the_sgx_server
```
For example: <br>
`./run_docker_local_example.sh /home/user/keys /home/user/password 127.0.0.1`

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
