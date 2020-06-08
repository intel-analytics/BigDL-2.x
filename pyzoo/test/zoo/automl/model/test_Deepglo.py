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

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.model.DTCNMF.DTCNMF_pytorch import DTCNMFPytorch
from numpy.testing import assert_array_almost_equal


class TestVanillaLSTM(ZooTestCase):

    def load_mat():
        Ymats=dict()
        with open('/home/ding/proj/deepglo/datasets/ymat_asiainfo.npy', 'rb') as f:
            for kpi_name in ['kpi0','kpi1','kpi2','kpi3','kpi4','kpi5','kpi6','kpi7']:
                Ymats[kpi_name] = np.load(f)
        return Ymats

    def setup_method(self, method):
        self.config = {
            'y_iters': 1,
            "init_epochs": 1,
            "max_FX_epoch": 1,
            "max_TCN_epoch": 1
        }
        self.model = DTCNMFPytorch()
        Ymats = load_mat()
        self.Ymat = Ymats["kpi2"]

    def teardown_method(self, method):
        pass

    def test_fit_eval(self):
        print("fit_eval:", self.model.fit_eval(self.Ymat,
                                               **self.config))

    def test_evaluate(self):
        pass

    def test_predict(self):
        pass

    def test_save_restore(self):
        pass

if __name__ == '__main__':
    pytest.main([__file__])
