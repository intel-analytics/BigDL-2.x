# from bigdl.transform.vision.image import *

import pytest
import os
from bigdl.util.common import *
from transform.vision.image import *


class TestLayer():

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf()
        self.sc = SparkContext(master="local[4]", appName="test model",
                               conf=sparkConf)
        init_engine()
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        self.image_path = os.path.join(resource_path, "image/000025.jpg")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def transformer_test(self, transformer):
        image_frame = ImageFrame.read(self.image_path)
        transformer(image_frame)
        image_frame.transform(transformer)
        image_frame.to_sample()

        image_frame = ImageFrame.read(self.image_path, self.sc)
        transformer(image_frame)
        image_frame.transform(transformer)
        image_frame.to_sample()

    def test_colorjitter(self):
        color = ColorJitter(random_order_prob=1.0, shuffle=True)
        self.transformer_test(color)

    def test_

if __name__ == "__main__":
    pytest.main([__file__])
