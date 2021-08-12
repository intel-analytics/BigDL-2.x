# Java Spark Occlum

## Spark 2.4.6 local test

Configure environment variables in `Dockerfile` and `build-docker-image.sh`.

Build the docker image:
``` bash
bash build-docker-image.sh
```

## How to Run

### Prerequisite
To launch Trusted Big Data ML applications on Graphene-SGX, you need to install graphene-sgx-driver:
```bash
sudo bash ../../../scripts/install-graphene-driver.sh
```

To train a model with PPML in Analytics Zoo and BigDL, you need to prepare the data first. The Docker image is taking lenet and MNIST as example.

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>

### Prepare the keys
The ppml in analytics zoo needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
sudo ../../../scripts/generate-keys.sh
```

You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.
It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.
```bash
openssl genrsa -3 -out enclave-key.pem 3072
```
### Prepare the password
Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file:

```bash
sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
```

To run Spark pi example, start the docker container with:
``` bash
bash start-spark-local.sh test
```
To run BigDL example, start the docker container with:
``` bash
bash start-spark-local.sh bigdl
```
The examples are run in the docker container. Attach it and see the results.

