#!/bin/bash
graphene-sgx ./bash -c "python ./work/examples/helloworld.py" | tee test-helloworld-sgx.log
