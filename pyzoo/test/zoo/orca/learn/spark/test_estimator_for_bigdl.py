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

import shutil
from unittest import TestCase

from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.pipeline.nnframes import *
from zoo.feature.common import *
from zoo.orca.learn.bigdl import Estimator
from bigdl.optim.optimizer import *


class TestEstimatorForKeras(TestCase):
    def get_estimator_df(self):
        self.sc = init_nncontext()
        data = self.sc.parallelize([
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0)),
            ((2.0, 1.0), (1.0, 2.0)),
            ((1.0, 2.0), (2.0, 1.0))])

        schema = StructType([
            StructField("features", ArrayType(DoubleType(), False), False),
            StructField("label", ArrayType(DoubleType(), False), False)])
        self.sqlContext = SQLContext(self.sc)
        df = self.sqlContext.createDataFrame(data, schema)
        return df

    def test_nnEstimator(self):
        linear_model = Sequential().add(Linear(2, 2))
        mse_criterion = MSECriterion()
        df = self.get_estimator_df()
        est = Estimator.from_bigdl(model=linear_model, loss=mse_criterion,
                                   feature_preprocessing=SeqToTensor([2]),
                                   label_preprocessing=SeqToTensor([2]))
        est.fit(df, 1, batch_size=4, optimizer=Adam())
        nn_model = est.get_model()
        res1 = nn_model.transform(df)
        res2 = est.predict(df)
        res1_c = res1.collect()
        res2_c = res2.collect()
        assert type(res1).__name__ == 'DataFrame'
        assert type(res2).__name__ == 'DataFrame'
        assert len(res1_c) == len(res2_c)
        for idx in range(len(res1_c)):
            assert res1_c[idx]["prediction"] == res2_c[idx]["prediction"]
        with tempfile.TemporaryDirectory() as tempdirname:
            temp_path = os.path.join(tempdirname, "model")
            est.save(temp_path)
            est2 = Estimator.from_bigdl(model=linear_model, loss=mse_criterion)
            est2.load(temp_path, optimizer=Adam(), loss=mse_criterion,
                      feature_preprocessing=SeqToTensor([2]), label_preprocessing=SeqToTensor([2]))
            res3 = est2.predict(df)
            res3_c = res3.collect()
            assert type(res3).__name__ == 'DataFrame'
            assert len(res1_c) == len(res3_c)
            for idx in range(len(res1_c)):
                assert res1_c[idx]["prediction"] == res3_c[idx]["prediction"]
            est2.fit(df, 1, batch_size=4, optimizer=Adam())


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
