GRAPHENEDIR ?= /home/sdp/qiyuan/redis_test/graphene
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem

G_JAVA_XMX ?= 2G
G_SGX_SIZE ?= 8G
G_SGX_THREAD_NUM ?= 256

THIS_DIR ?= /ppml/trusted-realtime-ml/java
JDK_HOME ?= /opt/jdk8
WORK_DIR ?= $(THIS_DIR)/work
FLINK_HOME ?= 

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

ifeq ($(SGX),)
GRAPHENE = graphene-direct
else
GRAPHENE = graphene-sgx
endif


.PHONY: all
all: java.manifest | pal_loader
ifeq ($(SGX),1)
all: java.token
endif

include $(GRAPHENEDIR)/Scripts/Makefile.configs

#### java
java.manifest: java.manifest.template
	graphene-manifest \
		-Dlog_level=$(GRAPHENE_LOG_LEVEL) \
		-Dexecdir=$(shell dirname $(shell which bash)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Djdk_home=$(JDK_HOME) \
		-Dspark_local_ip=$(SPARK_LOCAL_IP) \
        -Dspark_user=$(SPARK_USER) \
        -Dspark_home=$(SPARK_HOME) \
        -Dwork_dir=$(WORK_DIR) \
		-Dflink_home=$(FLINK_HOME) \
		-Dg_sgx_size=$(G_SGX_SIZE)
		$< > $@


java.manifest.sgx: java.manifest
	graphene-sgx-sign \
                -libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                -key $(SGX_SIGNER_KEY) \
                -manifest java.manifest -output $@

java.sig: java.manifest.sgx

java.token: java.sig
	graphene-sgx-get-token \
                -output java.token -sig java.sig

pal_loader:
	ln -s $(GRAPHENEDIR)/Runtime/pal_loader $@

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	./pal_loader ./bash -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/6 ]"
	@rm OUTPUT

	./pal_loader ./bash -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/6 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/6 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/6 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/6 ]"
	@rm OUTPUT

	./pal_loader ./bash -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/6 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig trusted-libs pal_loader OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
