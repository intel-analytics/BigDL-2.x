# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sgwhat/analytics-zoo/blob/colab/docs/docs/colab-notebook/orca/examples/fashion_mnist_bigdl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
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


# ## **Environment Preparation**
# 
# **Install Java 8**
# 
# Run the cell on the **Google Colab** to install jdk 1.8.
# 
# **Note:** if you run this notebook on your computer, root permission is required when running the cell to install Java 8. (You may ignore this cell if Java 8 has already been set up in your computer).

# In[ ]:


# Install jdk8
#.system('apt-get install openjdk-8-jdk-headless -qq > /dev/null')
#import os
# Set environment variable JAVA_HOME.
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
#.system('update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java')
#.system('java -version')


# **Install Analytics Zoo**
# 
# [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is needed to prepare the Python environment for running this example. 
# 
# **Note**: The following code cell is specific for setting up conda environment on Colab; for general conda installation, please refer to the [install guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for more details.

# In[ ]:


#import sys

# Get current python version
#version_info = sys.#version_info
#python_version = f"{#version_info.major}.{#version_info.minor}.{#version_info.micro}"


# In[ ]:


# Install Miniconda
#.system('wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh')
#.system('chmod +x Miniconda3-4.5.4-Linux-x86_64.sh')
#.system('./Miniconda3-4.5.4-Linux-x86_64.sh -b -f -p /usr/local')

# Update Conda
#.system('conda install --channel defaults conda python=$#python_version --yes')
#.system('conda update --channel defaults --all --yes')

# Append to the sys.path
#_ = (sys.path
#        #.append(f"/usr/local/lib/python{#version_info.major}.{#version_info.minor}/site-packages"))

#os.environ['PYTHONHOME']="/usr/local"


# You can install the latest pre-release version using `pip install --pre  analytics-zoo`.

# In[ ]:


# Install latest pre-release version of Analytics Zoo 
# Installing Analytics Zoo from pip will automatically install pyspark, bigdl, and their dependencies.
#.system('pip install --pre --upgrade analytics-zoo[ray]')


# In[ ]:


# Install python dependencies
#.system('pip install torch==1.7.1 torchvision==0.8.2')
#.system('pip install -U ray')
#.system('pip install jep==3.9.0')
#.system('pip install six cloudpickle')
#.system('pip install argparse')
#.system('pip install tensorboard')


# ## **BigDL using Orca APIs**
# 
# In this guide we will describe how to scale out PyTorch programs using Orca in 4 simple steps.

# In[ ]:


#import necessary libraries and modules
#from __future__ import print_function
import numpy as np

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca import OrcaContext


# ### **Step 1: Init Orca Context**
# 

# In[ ]:


# recommended to set it to True when running Analytics Zoo in Jupyter notebook. 
OrcaContext.log_output = True # (this will display terminal's stdout and stderr in the Jupyter notebook).

cluster_mode = "local"

if cluster_mode == "local":
    init_orca_context(cores=1, memory="2g") # run in local mode
elif cluster_mode == "k8s":
    init_orca_context(cluster_mode="k8s", num_nodes=2, cores=4) # run on K8s cluster
elif cluster_mode == "yarn":
    init_orca_context(
        cluster_mode="yarn-client", cores=4, num_nodes=2, memory="2g",
        driver_memory="10g", driver_cores=1,
        conf={"spark.rpc.message.maxSize": "1024",
              "spark.task.maxFailures": "1",
              "spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"}) # run on Hadoop YARN cluster


# This is the only place where you need to specify local or distributed mode. View [Orca Context](https://analytics-zoo.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) for more details.
# 
# **Note**: You should export HADOOP_CONF_DIR=/path/to/hadoop/conf/dir when you run on Hadoop YARN cluster.

# ### **Step 2: Define the Model**
# You may define your model, loss and optimizer in the same way as in any standard (single node) PyTorch program.

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# After defining your model, you need to define a *Model Creator Function* that returns an instance of your model, and a *Optimizer Creator Function* that returns a pytorch optimizer.

# In[ ]:


import torch.optim as optim

def model_creator(config):
    model = Net()
    return model

def optimizer_creator(model, config):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return optimizer


# ### **Step 3: Define Train Dataset**
# 
# You can define the dataset using a *Data Creator Function* that has two parameters `config` and `batch_size`.

# In[ ]:


# training loss vs. epochs
criterion = nn.CrossEntropyLoss()
batch_size = 320
epochs = 5


# In[ ]:


import torchvision
import torchvision.transforms as transforms

def train_data_creator(config, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader

def validation_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return testloader


# ### **Step 4: Using TensorBoard in PyTorch**
# 
# Now using TensorBoard with PyTorch!

# In[ ]:


import matplotlib.pyplot as plt

# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# Before logging anything, we need to create a SummaryWriter instance.

# In[ ]:


# Load the TensorBoard notebook extension
#.run_line_magic('load_ext', 'tensorboard')


# A brief overview of the dashboards shown (tabs in top navigation bar):
# 
# * The **Scalars** dashboard shows how the loss and metrics change with every epoch. You can use it to also track training speed, learning rate, and other scalar values.
# * The **Graphs** dashboard helps you visualize your model. In this case, the graph of layers is shown which can help you ensure it is built correctly. 

# In[ ]:


from torch.utils.tensorboard import SummaryWriter

tensorboard_dir = "runs"
writer = SummaryWriter(tensorboard_dir + '/bigdl')
# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# plot some random training images
dataiter = iter(train_data_creator(config={}, batch_size=4))
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

# inspect the model using tensorboard
writer.add_graph(model_creator(config={}), images)


# If you do not need the summary writer anymore, call close() method.

# In[ ]:


writer.close()


# ### **Step 5: Fit with Orca Estimator**
# 
# First, Create an Estimator and set its backend to `bigdl`.
# 

# In[ ]:


train_loader = train_data_creator(config={}, batch_size=batch_size)
test_loader = validation_data_creator(config={}, batch_size=batch_size)
net = model_creator(config={})
optimizer = optimizer_creator(model=net, config={"lr": 0.001})


# First, Create an Estimator.

# In[ ]:


from zoo.orca.learn.pytorch import Estimator
from zoo.orca.learn.metrics import Accuracy

orca_estimator = Estimator.from_torch(model=net, optimizer=optimizer, loss=criterion, metrics=[Accuracy()], backend="bigdl")


# In[ ]:


orca_estimator.set_tensorboard(tensorboard_dir, "bigdl")


# Next, fit using the Estimator.

# In[ ]:


from zoo.orca.learn.trigger import EveryEpoch

orca_estimator.fit(data=train_loader, epochs=epochs, validation_data=test_loader, checkpoint_trigger=EveryEpoch())


# Finally, evaluate using the Estimator. 

# In[ ]:


res = orca_estimator.evaluate(data=test_loader)
print("Accuracy of the network on the test images: %s" % res)


# The accuracy of this model has reached 70%.

# In[ ]:


# stop orca context when program finishes
stop_orca_context()


# ### **Step 6: Visualization by Tensorboard**
# 
# TensorBoard is a visualization toolkit for machine learning experimentation. TensorBoard allows tracking and visualizing metrics such as loss and accuracy, visualizing the model graph, viewing histograms, displaying images and much more.

# Start TensorBoard, specifying the root log directory you used above. 
# Argument ``logdir`` points to directory where TensorBoard will look to find 
# event files that it can display. TensorBoard will recursively walk 
# the directory structure rooted at logdir, looking for .*tfevents.* files.

# In[ ]:


# move into logdata file for tensorboard using
#.run_line_magic('tensorboard', "--logdir '/content/runs/bigdl'")


# This dashboard shows how the loss change with every epoch.

# In[ ]:


from tensorboard import notebook
# View open TensorBoard instances
notebook.list()

