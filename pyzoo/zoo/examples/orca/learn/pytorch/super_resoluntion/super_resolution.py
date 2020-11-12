from __future__ import print_function
import argparse
from math import ceil
from os import listdir
from os.path import join
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from zoo.orca.learn.pytorch import Estimator
from zoo.orca import init_orca_context, stop_orca_context


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.image_filenames)


def get_training_set(upscale_factor, dir):
    root_dir = join(dir, "BSDS300/images")
    print("root dir is " + root_dir)
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=Compose([
                                 CenterCrop(crop_size),
                                 Resize(crop_size // upscale_factor),
                                 ToTensor()]),
                             target_transform=Compose([
                                 CenterCrop(crop_size),
                                 ToTensor()]))


def get_test_set(upscale_factor, dir):
    root_dir = join(dir, "BSDS300/images")
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=Compose([
                                 CenterCrop(crop_size),
                                 Resize(crop_size // upscale_factor),
                                 ToTensor()]),
                             target_transform=Compose([
                                 CenterCrop(crop_size),
                                 ToTensor()]))


class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def train_data_creator(config):
    train_set = get_training_set(config.get("upscale_factor", 3), config.get("dataset", "dataset"))
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=4,
                                      batch_size=config.get('batchSize', 64),
                                      shuffle=True)
    return training_data_loader


def validation_data_creator(config):
    print('===> Loading datasets')
    test_set = get_test_set(config.get("upscale_factor", 3), config.get("dataset", "dataset"))
    testing_data_loader = DataLoader(dataset=test_set,
                                     num_workers=4,
                                     batch_size=config.get("testBatchSize", 100),
                                     shuffle=False)
    return testing_data_loader


def model_creator(config):
    net = Net(upscale_factor=config.get("upscale_factor", 3))
    net.train()
    return net


def Adam_creator(model, config):
    return optim.Adam(model.parameters(), lr=config.get("lr", 0.001))


def train(opt):
    if opt.mode == "local":
        init_orca_context(cores=1, memory="20g")
    elif opt.mode == "yarn":
        init_orca_context(
            cluster_mode="yarn-client", cores=4, num_nodes=opt.node, memory="2g",
            driver_memory="10g", driver_cores=1,
            conf={"spark.rpc.message.maxSize": "1024",
                  "spark.task.maxFailures": "1"
                  },
            additional_archive="dataset.zip#dataset")
    else:
        print("init orca context failed")

    print('===> Building model')
    node_number=1 if opt.mode =="local" else opt.node
    estimator = Estimator.from_torch(
        model=model_creator,
        optimizer=Adam_creator,
        loss=nn.MSELoss(),
        backend="pytorch",
        config={
            "lr": opt.lr,
            "upscale_factor": opt.upscale_factor,
            "batchSize": ceil(opt.batchSize / node_number),
            "testBatchSize": ceil(opt.testBatchSize / node_number),
            "nEpochs": opt.nEpochs,
            "dataset": opt.dataset,
            "mode": opt.mode
        }
    )

    stats = estimator.fit(data=train_data_creator, epochs=opt.nEpochs)
    val_stats = estimator.evaluate(data=validation_data_creator)
    for epochinfo in stats:
        print("train stats==> num_samples:" + str(epochinfo["num_samples"])
              + " ,epoch:"+ str(epochinfo["epoch"])
              + " ,batch_count:" + str(epochinfo["batch_count"])
              + " ,train_loss" + str(epochinfo["train_loss"])
              + " ,last_train_loss" + str(epochinfo["last_train_loss"]))
    print("validation stats==> num_samples:"+str(val_stats["num_samples"])
          + " ,batch_count:" + str(val_stats["batch_count"])
          + " ,val_loss" + str(val_stats["val_loss"])
          + " ,last_val_loss" + str(val_stats["last_val_loss"]))

    stop_orca_context()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--mode', type=str, default='local', help='The mode of spark cluster.')
    parser.add_argument('--dataset', type=str, default='./dataset', help='The dir of dataset.')
    parser.add_argument('--node', type=int, default=1, help='number of working nodes')
    opt = parser.parse_args()

    print(opt)
    train(opt)
