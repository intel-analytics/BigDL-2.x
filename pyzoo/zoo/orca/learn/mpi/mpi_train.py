#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import cloudpickle
from zoo.util.utils import get_node_ip

print("Worker on {} with global rank {}".format(get_node_ip(), os.environ.get("PMI_RANK", 0)))

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', type=str, default="",
                    help='The directory of the pkl files for mpi training.')
args = parser.parse_args()
pkl_path = args.pkl_path

with open("{}/saved_mpi_estimator.pkl".format(pkl_path), "rb") as f:
    model_creator, optimizer_creator, loss_creator, \
        scheduler_creator, config, init_func = cloudpickle.load(f)

with open("{}/mpi_train_data.pkl".format(pkl_path), "rb") as f:
    train_data_creator, epochs, validation_data_creator, train_func, \
        validate_func, train_batches, validate_batches, validate_steps = cloudpickle.load(f)

if init_func:
    print("Initializing distributed environment")
    init_func()

# Wrap DDP should be done by users in model_creator
model = model_creator(config)
optimizer = optimizer_creator(model, config)
loss = loss_creator  # assume it is an instance
scheduler = scheduler_creator(optimizer, config)
train_ld = train_data_creator(config)
train_batches = train_batches if train_batches else len(train_ld)
print("Batches to train: ", train_batches)
if validation_data_creator:
    valid_ld = validation_data_creator(config)
    validate_batches = validate_batches if validate_batches else len(valid_ld)
    print("Batches to test: ", validate_batches)

for i in range(epochs):
    train_func(model, train_ld, train_batches, optimizer, loss, scheduler, config)
    if validation_data_creator:
        validate_func(model, valid_ld, validate_batches, config)

# train_ld = train_data_creator(config)
# train_batches = train_batches if train_batches else len(train_ld)
# print("Batches to train: ", train_batches)
# train_iter = iter(train_ld)
# for j in range(train_batches):
#     if j > 0 and j % len(train_ld) == 0:  # For the case where there are not enough batches.
#         train_iter = iter(train_ld)
#     x, y = next(train_iter)
    # print("X_int ", x[0].shape)
    # print("X_cat ", x[1].shape)
    # print("y ", y.shape)
