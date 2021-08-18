import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from bigdl.nano.pytorch.accelerators.ipex_accelerator import IPEXAccelerator
from test._train_torch_lightning import train_with_linear_top_layer


batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestModelsVision(TestCase):

    def test_resnet18_ipex(self):
        resnet18 = vision.resnet18(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_resnet34_ipex(self):
        resnet34 = vision.resnet34(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet34, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_resnet50_ipex(self):
        resnet50 = vision.resnet50(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet50, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_mobilenet_v3_large_ipex(self):
        mobilenet = vision.mobilenet_v3_large(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_mobilenet_v3_small_ipex(self):
        mobilenet = vision.mobilenet_v3_small(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_mobilenet_v2_ipex(self):
        mobilenet = vision.mobilenet_v2(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())

    def test_shufflenet_ipex(self):
        shufflenet = vision.shufflenet_v2_x1_0(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            shufflenet, batch_size, num_workers, data_dir,
            accelerator=IPEXAccelerator())


if __name__ == '__main__':
    pytest.main([__file__])
