import argparse
import cloudpickle
from zoo.util.utils import get_node_ip

parser = argparse.ArgumentParser()
parser.add_argument('--pkl_path', type=str, default="", help='The directory of the pkl files for mpi training.')
args = parser.parse_args()
pkl_path = args.pkl_path

with open("{}/saved_mpi_estimator.pkl".format(pkl_path), "rb") as f:
    model_creator, optimizer_creator, loss_creator, scheduler_creator, config, init_func = cloudpickle.load(f)

with open("{}/mpi_train_data.pkl".format(pkl_path), "rb") as f:
    train_data_creator, epochs, validation_data_creator, train_func, validate_func, train_steps, validate_steps = cloudpickle.load(f)

if init_func:
    print("Initializing distributed environment on ", get_node_ip())
    init_func()

# Wrap DDP should be done by users in model_creator
# model = model_creator(config)
# optimizer = optimizer_creator(model, config)
# loss = loss_creator  # assume it is an instance
# scheduler = scheduler_creator(optimizer, config)
# train_ld = train_data_creator(config)
# train_batches = train_steps if train_steps else len(train_ld)
# print("Batches to train: ", train_batches)
# if validation_data_creator:
#     valid_ld = validation_data_creator(config)
#     validate_batches = validate_steps if validate_steps else len(valid_ld)
#     print("Batches to test: ", validate_batches)
#
# for i in range(epochs):
#     train_func(model, train_ld, train_batches, optimizer, loss, scheduler)
#     if validation_data_creator:
#         validate_func(model, valid_ld, validate_batches)

train_ld = train_data_creator(config)
train_batches = train_steps if train_steps else len(train_ld)
print("Batches to train: ", train_batches)
train_iter = iter(train_ld)
for j in range(train_batches):
    if j > 0 and j % len(train_ld) == 0:  # For the case where there are not enough batches.
        train_iter = iter(train_ld)
    x, y = next(train_iter)
