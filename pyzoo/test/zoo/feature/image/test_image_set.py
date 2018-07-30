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

import pytest
import cv2

from zoo.common.nncontext import *
from zoo.feature.image import *


class Test_Image_Set():

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.sc = init_nncontext(init_spark_conf().setMaster("local[4]")
                                 .setAppName("test image set"))
        resource_path = os.path.join(os.path.split(__file__)[0], "../../resources")
        self.image_path = os.path.join(resource_path, "pascal/000025.jpg")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def transformer_test(self, transformer):
        image_set = ImageSet.read(self.image_path)
        transformed = transformer(image_set)
        transformed.get_image()

        image_set = ImageSet.read(self.image_path, self.sc)
        transformed = transformer(image_set)
        images = transformed.get_image()
        images.count()

    def test_get_image(self):
        image_set = ImageSet.read(self.image_path, resize_height=128, resize_width=128)
        image_set.get_image()

    def test_get_label(self):
        image_set = ImageSet.read(self.image_path)
        image_set.get_label()

    def test_is_local(self):
        image_set = ImageSet.read(self.image_path)
        assert image_set.is_local() is True
        image_set = ImageSet.read(self.image_path, self.sc)
        assert image_set.is_local() is False

    def test_is_distributed(self):
        image_set = ImageSet.read(self.image_path)
        assert image_set.is_distributed() is False
        image_set = ImageSet.read(self.image_path, self.sc)
        assert image_set.is_distributed() is True

    def test_image_set_transform(self):
        transformer = ImageMatToTensor()
        image_set = ImageSet.read(self.image_path)
        transformed = image_set.transform(transformer)
        transformed.get_image()

    def test_empty_get_predict_local(self):
        image_set = ImageSet.read(self.image_path)
        image_set.get_predict()

    def test_empty_get_predict_distributed(self):
        image_set = ImageSet.read(self.image_path, self.sc)
        image_set.get_predict()

    def test_local_image_set(self):
        image = cv2.imread(self.image_path)
        local_image_set = LocalImageSet([image])
        print(local_image_set.get_image())

if __name__ == "__main__":
    pytest.main([__file__])
