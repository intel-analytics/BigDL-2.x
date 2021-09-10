#!/bin/bash
graphene-sgx ./bash -c "python ./work/examples/test-numpy.py" | tee test-numpy-sgx.log
