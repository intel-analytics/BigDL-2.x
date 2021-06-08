manifests = $(addsuffix .manifest,label_image)
exec_target = $(manifests)
target = label_image

GRAPHENEDIR ?= ../..
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem
TF_DIR ?= tensorflow
BAZEL_BIN ?= $(HOME)/bin/bazel

GIT_COMMIT ?= v1.9.0
TAR_SHA256 ?= ffc3151b06823d57b4a408261ba8efe53601563dfe93af0866751d4f6ca5068c

ifeq ($(DEBUG),1)
GRAPHENE_LOG_LEVEL = debug
else
GRAPHENE_LOG_LEVEL = error
endif

.PHONY: all
all: label_image.manifest
ifeq ($(SGX),1)
all: label_image.manifest.sgx label_image.sig label_image.token
endif

ifeq ($(SGX),)
GRAPHENE = graphene-direct
else
GRAPHENE = graphene-sgx
endif

include ../../Scripts/Makefile.configs

$(TF_DIR)/configure:
	$(GRAPHENEDIR)/Scripts/download --output tensorflow.tar.gz --sha256 $(TAR_SHA256)\
		--url https://github.com/tensorflow/tensorflow/archive/$(GIT_COMMIT).tar.gz
	mkdir $(TF_DIR)
	tar -C $(TF_DIR) --strip-components=1 -xf tensorflow.tar.gz

$(TF_DIR)/bazel-bin/tensorflow/contrib/lite/examples/label_image/label_image: $(TF_DIR)/configure
	cd $(TF_DIR) && $(BAZEL_BIN) build tensorflow/contrib/lite/examples/label_image

label_image: $(TF_DIR)/bazel-bin/tensorflow/contrib/lite/examples/label_image/label_image
	cp $^ .

libtensorflow_framework.so: label_image
	cp $(TF_DIR)/bazel-bin/tensorflow/libtensorflow_framework.so $@

INCEPTION_HASH=b1a1f91276e48a9ddf0cb0d854f044ebfbf985dc2c2cedceb52b3d668894299a
inception_v3.tflite:
	$(GRAPHENEDIR)/Scripts/download --output inception_v3_2018_04_27.tgz --sha256 $(INCEPTION_HASH)\
		--url https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
	tar xfz inception_v3_2018_04_27.tgz

labels.txt: $(TF_DIR)/tensorflow/contrib/lite/java/ovic/src/testdata/labels.txt
	cp $^ $@

image.bmp: $(TF_DIR)/tensorflow/contrib/lite/examples/label_image/testdata/grace_hopper.bmp
	cp $^ $@

$(GRAPHENEDIR)/Runtime/libgcc_s.so.1:
	cp $(ARCH_LIBDIR)/libgcc_s.so.1 $@

$(GRAPHENEDIR)/Runtime/libstdc++.so.6:
	cp /usr/$(ARCH_LIBDIR)/libstdc++.so.6 $@

label_image.manifest: label_image.manifest.template libtensorflow_framework.so label_image \
                      inception_v3.tflite labels.txt image.bmp
	graphene-manifest \
		-Dlog_level=$(GRAPHENE_LOG_LEVEL) \
		$< > $@

label_image.manifest.sgx: label_image.manifest inception_v3.tflite labels.txt
	graphene-sgx-sign \
		--key $(SGX_SIGNER_KEY) \
		--manifest $< \
		--output $@
	graphene-sgx-get-token \
		--output label_image.token --sig label_image.sig

.PHONY: check
check: all
	$(GRAPHENE) ./label_image -m inception_v3.tflite -i image.bmp -t 4

.PHONY: run-native
run-native: all
	./label_image -m inception_v3.tflite -i image.bmp -t 4

.PHONY: run-graphene
run-graphene: all
	$(GRAPHENE) ./label_image -m inception_v3.tflite -i image.bmp -t 4

.PHONY: clean
clean:
	$(RM) label_image.manifest label_image.manifest.sgx label_image.sig label_image.token

.PHONY: distclean
distclean: clean
	$(RM) -r label_image
	$(RM) inception_v3_2018_04_27.tgz inception_v3.pb inception_v3.tflite labels.txt image.bmp
	$(RM) -r $(TF_DIR) tensorflow.tar.gz libtensorflow_framework.so bazel-*-installer-linux-*.sh

BAZEL_INSTALLER_HASH=17ab70344645359fd4178002f367885e9019ae7507c9c1ade8220f3628383444
.PHONY: install-dependencies-ubuntu
install-dependencies-ubuntu:
	apt-get update
	apt-get install -y python-dev python-pip wget git
	# https://docs.bazel.build/versions/master/install-ubuntu.html
	apt-get install -y pkg-config zip g++ zlib1g-dev unzip python
	$(GRAPHENEDIR)/Scripts/download --output bazel-0.16.1-installer-linux-x86_64.sh --sha256 $(BAZEL_INSTALLER_HASH)\
		--url https://github.com/bazelbuild/bazel/releases/download/0.16.1/bazel-0.16.1-installer-linux-x86_64.sh
	chmod +x bazel-0.16.1-installer-linux-x86_64.sh
	./bazel-0.16.1-installer-linux-x86_64.sh --user

.PHONY: install-dependencies-fedora
install-dependencies-fedora:
	dnf -y install python3-devel python3-pip wget git pkg-config zip gcc-g++ zlib-devel unzip
	$(GRAPHENEDIR)/Scripts/download --output bazel-0.16.1-installer-linux-x86_64.sh --sha256 $(BAZEL_INSTALLER_HASH)\
		--url https://github.com/bazelbuild/bazel/releases/download/0.16.1/bazel-0.16.1-installer-linux-x86_64.sh
	chmod +x bazel-0.16.1-installer-linux-x86_64.sh
	./bazel-0.16.1-installer-linux-x86_64.sh --user
