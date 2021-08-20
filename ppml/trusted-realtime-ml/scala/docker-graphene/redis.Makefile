GRAPHENEDIR ?= graphene-sgx
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem
ARCH_LIBDIR ?= /lib/$(shell $(CC) -dumpmachine)

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

#include $(GRAPHENEDIR)/Scripts/Makefile.configs

bash.manifest: bash.manifest.template
	graphene-manifest \
		-Dexecdir=$(shell dirname $(shell which bash)) \
		-Darch_libdir=$(ARCH_LIBDIR) \
		-Dwork_dir=$(WORK_DIR) \
		-Dg_sgx_size=$(G_SGX_SIZE) \
		$< >$@


bash.sig bash.manifest.sgx &: bash.manifest src/src/redis-server
	graphene-sgx-sign \
                -libpal $(GRAPHENEDIR)/Runtime/libpal-Linux-SGX.so \
                -key $(SGX_SIGNER_KEY) \
                -manifest bash.manifest \
                -output bash.manifest.sgx

bash.token: bash.sig
	graphene-sgx-get-token -output $@ -sig $<



.PHONY: start-native-server
start-native-server: all
	./redis-server --save '' --protected-mode no

.PHONY: start-graphene-server
start-graphene-server: all
	graphene-sgx ./bash -c "redis-server --save '' --protected-mode no"

.PHONY: clean
clean:
	$(RM) *.token *.sig *.manifest.sgx *.manifest *.rdb
