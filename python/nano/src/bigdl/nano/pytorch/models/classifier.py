from typing import Callable

import pytorch_lightning as pl
import torch
import torch.nn as nn
from bigdl.nano.pytorch.models._utils import BackboneModule


class ImageClassifier(pl.LightningModule):
    # Linear top layer

    def __init__(self, backbone: BackboneModule,
                 num_classes: int,
                 head: nn.Module = None,
                 criterion: nn.Module = None):
        super().__init__()

        if head is None:
            output_size = backbone.get_output_size()
            head = nn.Linear(output_size, num_classes)
        self.model = torch.nn.Sequential(backbone, head)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.criterion(output, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
