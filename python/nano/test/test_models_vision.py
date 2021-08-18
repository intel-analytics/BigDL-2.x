import pytest
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test._train_torch_lightning import train_with_linear_top_layer


batch_size = 256
num_workers = 0
data_dir = "./data"


class TestModelsVision(TestCase):

    def test_resnet18(self):
        resnet18 = vision.resnet18(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet18, batch_size, num_workers, data_dir)

    def test_resnet34(self):
        resnet34 = vision.resnet34(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet34, batch_size, num_workers, data_dir)

    def test_resnet50(self):
        resnet50 = vision.resnet50(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            resnet50, batch_size, num_workers, data_dir)

    def test_mobilenet_v3_large(self):
        mobilenet = vision.mobilenet_v3_large(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir)

    def test_mobilenet_v3_small(self):
        mobilenet = vision.mobilenet_v3_small(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir)

    def test_mobilenet_v2(self):
        mobilenet = vision.mobilenet_v2(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            mobilenet, batch_size, num_workers, data_dir)

    def test_shufflenet(self):
        shufflenet = vision.shufflenet_v2_x1_0(
            pretrained=True, include_top=False, freeze=True)
        train_with_linear_top_layer(
            shufflenet, batch_size, num_workers, data_dir)


if __name__ == '__main__':
    pytest.main([__file__])
