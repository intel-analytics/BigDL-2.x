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

import numpy as np
import pandas as pd
import os
from numpy.testing import assert_array_almost_equal

from zoo.automl.model.XGBoost import XGBoost
from zoo.automl.feature.identity_transformer import IdentityTransformer


class TestXgbregressor(ZooTestCase):
    def setup_method(self, method):
        # super().setup_method(method)
        self.model = XGBoost(config={'n_estimators': 5, 'max_depth': 2, 'tree_method': 'hist'})
        feature_cols = ["f", "f2"]
        target_col = "t"
        train_df = pd.DataFrame({"f": np.random.randn(20),
                                 "f2": np.random.randn(20),
                                 "t": np.random.randint(20)})
        val_df = pd.DataFrame({"f": np.random.randn(5),
                               "f2": np.random.randn(5),
                               "t": np.random.randint(5)})

        ft = IdentityTransformer(feature_cols=feature_cols, target_col=target_col)

        self.x, self.y = ft.transform(train_df)
        self.val_x, self.val_y = ft.transform(val_df)

    def teardown_method(self, method):
        pass

    def test_fit_predict_evaluate(self):
        self.model.fit_eval(self.x, self.y, [(self.val_x, self.val_y)])

        # test predict
        result = self.model.predict(self.val_x)

        # test evaluate
        evaluate_result = self.model.evaluate(self.val_x, self.val_y)

    def test_save_restore(self):
        self.model.fit_eval(self.x, self.y, [(self.val_x, self.val_y)])

        result_save = self.model.predict(self.val_x)
        model_file = "tmp.pkl"
        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = XGBoost()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(self.val_x)
        assert_array_almost_equal(result_save, result_restore, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(result_save, result_restore)
        os.remove(model_file)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
