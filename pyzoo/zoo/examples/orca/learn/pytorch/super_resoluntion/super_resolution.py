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

from __future__ import print_function
import argparse
from PIL import Image
import urllib
import tarfile
from os import makedirs, remove, listdir
from os.path import exists, join, basename

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from zoo.orca.learn.pytorch import Estimator
from zoo.orca import init_orca_context, stop_orca_context



def download_report(count, block_size, total_size):
    downloaded = count * block_size
    percent = 100. * downloaded / total_size
    percent = min(100, percent)
    print('downloaded %d, %.2f%% completed' % (downloaded, percent))


def download_bsd300(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        file_path = join(dest, basename(url))
        urllib.request.urlretrieve(url, file_path, download_report)

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)

        remove(file_path)

    return output_image_dir


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


def get_training_set(upscale_factor, dir, isDownload):
    root_dir = download_bsd300(dir) if isDownload else join(dir, "BSDS300/images")
    print("root dir is " + root_dir)
    train_dir = join(root_dir, "train")
    crop_size = 256 - (256 % upscale_factor)

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
    crop_size = 256 - (256 % upscale_factor)

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
    train_set = get_training_set(config.get("upscale_factor", 3), config.get("dataset", "dataset"),
                                 config.get("download", False))
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


def optim_creator(model, config):
    return optim.Adam(model.parameters(), lr=config.get("lr", 0.001))


def train(opt):
    if opt.cluster_mode == "local":
        init_orca_context(cores=1, memory="20g")
    elif opt.cluster_mode == "yarn":
        additional = "" if opt.download else "BSDS300.zip#"+opt.dataset
        init_orca_context(
            cluster_mode="yarn-client", cores=opt.cores, num_nodes=opt.num_nodes, memory=opt.memory,
            driver_memory="10g", driver_cores=1,
            additional_archive=additional)
    else:
        print("init orca context failed")

    print('===> Building model')
    estimator = Estimator.from_torch(
        model=model_creator,
        optimizer=optim_creator,
        loss=nn.MSELoss(),
        backend="pytorch",
        config={
            "lr": opt.lr,
            "upscale_factor": opt.upscale_factor,
            "batchSize": opt.batchSize,
            "testBatchSize": opt.testBatchSize,
            "dataset": opt.dataset,
            "download": opt.download
        }
    )

    stats = estimator.fit(data=train_data_creator, epochs=opt.epochs)
    val_stats = estimator.evaluate(data=validation_data_creator)
    for epochinfo in stats:
        print("train stats==> num_samples:" + str(epochinfo["num_samples"])
              + " ,epoch:" + str(epochinfo["epoch"])
              + " ,batch_count:" + str(epochinfo["batch_count"])
              + " ,train_loss" + str(epochinfo["train_loss"])
              + " ,last_train_loss" + str(epochinfo["last_train_loss"]))
    print("validation stats==> num_samples:" + str(val_stats["num_samples"])
          + " ,batch_count:" + str(val_stats["batch_count"])
          + " ,val_loss" + str(val_stats["val_loss"])
          + " ,last_val_loss" + str(val_stats["last_val_loss"]))

    stop_orca_context()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--cluster_mode', type=str, default='local', help='The mode of spark cluster.')
    parser.add_argument("--num_nodes", type=int, default=1, help="The number of nodes to be used in the cluster. "
                                                                 "You can change it depending on your own cluster setting.")
    parser.add_argument("--cores", type=int, default=4,
                        help="The number of cpu cores you want to use on each node. ")
    parser.add_argument("--memory", type=str, default="2g",
                        help="The memory you want to use on each node. "
                             "You can change it depending on your own cluster setting.")
    parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
    parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
    parser.add_argument('--dataset', type=str, default='dataset', help='The dir of dataset.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
    parser.add_argument('--download', action="store_true", default=False, help="Auto download dataset.")
    opt = parser.parse_args()

    train(opt)
