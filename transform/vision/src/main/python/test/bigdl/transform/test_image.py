# from bigdl.transform.vision.image import *

import pytest
import os
import cv2
from transform.vision.image import *


class TestLayer():

    def get_image_feature(self):
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        image_path = os.path.join(resource_path, "image/000025.jpg")
        image = cv2.imread(image_path)
        return ImageFeature(image)

    def test_colorjitter(self):
        image = self.get_image_feature()
        color = ColorJitter(random_order_prob=1.0, shuffle=True)
        color.transform(image)

if __name__ == "__main__":
    pytest.main([__file__])
