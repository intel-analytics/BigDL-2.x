GRAPHENEDIR = /graphene
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

.PHONY: all
all: redis-server.manifest pal_loader
ifeq ($(SGX),1)
all: redis-server.manifest.sgx redis-server.sig redis-server.token
endif

#include $(GRAPHENEDIR)/Scripts/Makefile.configs

redis-server.manifest: redis-server.manifest.template
		-Dlog_level=$(GRAPHENE_LOG_LEVEL) \
		-Dexecdir=$(shell dirname $(shell which bash)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
        -Dwork_dir=$(WORK_DIR) \
		$< >$@


redis-server.sig redis-server.manifest.sgx &: redis-server.manifest src/src/redis-server
	graphene-sgx-sign \
                -libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                -key $(SGX_SIGNER_KEY) \
                -manifest redis-server.manifest \
                -output redis-server.manifest.sgx

redis-server.token: redis-server.sig
	graphene-sgx-get-token -output $@ -sig $<

pal_loader:
	ln -s $(GRAPHENEDIR)/Runtime/pal_loader $@

.PHONY: start-native-server
start-native-server: all
	./redis-server --save '' --protected-mode no

.PHONY: start-graphene-server
start-graphene-server: all
	./pal_loader redis-server --save '' --protected-mode no

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest pal_loader *.rdb
