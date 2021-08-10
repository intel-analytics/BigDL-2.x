SGX_SIGNER_KEY ?= /home/sdp/hangrui/myzoo/analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene/bashexm/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)
JDK_HOME ?= /home/sdp/opt/jdk

SPARK_LOCAL_IP ?= 127.0.0.1
SPARK_USER ?= root
SPARK_HOME ?= $(THIS_DIR)/work/spark-2.4.3/
WORK_DIR ?= $(THIS_DIR)/work
#SPARK_LOCAL_IP ?= $(THIS_DIR)/work/spark-2.4.6-bin-hadoop2.7
EXECDIR?=/home

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

.PHONY: all
all: bash.manifest
ifeq ($(SGX),1)
all: bash.manifest.sgx bash.sig bash.token
endif

bash.manifest: manifest.template
	graphene-manifest \
		-Dlog_level=$(GRAPHENE_LOG_LEVEL) \
		-Dexecdir=$(shell dirname $(shell which bash)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Djdk_home=$(JDK_HOME) \
		-Dspark_local_ip=$(SPARK_LOCAL_IP) \
        -Dspark_user=$(SPARK_USER) \
        -Dspark_home=$(SPARK_HOME) \
        -Dwork_dir=$(WORK_DIR) \
		$< >$@


#-Dexecdir=$(shell dirname $(shell which bash)) \

bash.manifest.sgx: bash.manifest
	@test -s $(SGX_SIGNER_KEY) || \
	    { echo "SGX signer private key was not found, please specify SGX_SIGNER_KEY!"; exit 1; }
	graphene-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
		--manifest bash.manifest \
		--output $@

bash.sig: bash.manifest.sgx

bash.token: bash.sig
	graphene-sgx-get-token --output bash.token --sig bash.sig

ifeq ($(SGX),)
GRAPHENE = graphene-direct
else
GRAPHENE = graphene-sgx
endif

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	$(GRAPHENE) ./bash -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/7 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/7 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/7 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/7 ]"
	@rm OUTPUT

	$(GRAPHENE) ./bash -c "readlink /proc/self/exe" > OUTPUT
	@grep -qE "^(/usr)?/bin/readlink" OUTPUT && echo "[ Success 7/7 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
