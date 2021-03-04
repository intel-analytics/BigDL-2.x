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
import torch


def _unify_input_formats(preds, target):
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds the same or one more dimensions than targets")

    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, dim=1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        preds = (preds >= 0.5).long()
    return preds, target


class Accuracy:

    def __init__(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

    def __call__(self, preds, targets):
        preds, target = _unify_input_formats(preds, targets)
        self.correct += torch.sum(torch.eq(preds, targets))
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class SparseCategoricalAccuracy:

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        if preds.ndim == targets.ndim:
            targets = torch.squeeze(targets, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds = preds.type_as(targets)
        self.correct += torch.sum(torch.eq(preds, targets))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class CategoricalAccuracy:
    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        self.correct += torch.sum(
            torch.eq(
                torch.argmax(preds, dim=-1), torch.argmax(targets, dim=-1)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class BinaryAccuracy:
    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets, threshold=0.5):
        threshold = torch.tensor(threshold)
        self.correct += torch.sum(
            torch.eq(
                torch.gt(preds, threshold), targets))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class Top5Accuracy:
    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        if preds.ndim == targets.ndim:
            targets = torch.squeeze(targets, dim=-1)
        batch_size = targets.size(0)

        _, preds = preds.topk(5, dim=-1, largest=True, sorted=True)
        preds = preds.type_as(targets).t()
        targets = targets.view(1, -1).expand_as(preds)

        self.correct += preds.eq(targets).view(-1).sum()
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total
