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
import numpy as np
from zoo.zouwu.model.forecast.tcmf_forecaster import TCMFForecaster
from unittest import TestCase
import tempfile


class TestZouwuModelTCMFForecaster(TestCase):

    def setUp(self):
        self.model = TCMFForecaster(y_iters=1,
                                    init_FX_epoch=1,
                                    max_FX_epoch=1,
                                    max_TCN_epoch=1,
                                    alt_iters=2)
        self.num_samples = 300
        self.horizon = np.random.randint(1, 50)
        seq_len = 480
        self.data = np.random.rand(self.num_samples, seq_len)
        self.id = np.arange(self.num_samples)
        self.data_new = np.random.rand(self.num_samples, self.horizon)

    def test_ndarray_local(self):
        ndarray_input = {'id': self.id, 'y': self.data}
        self.model.fit(ndarray_input)
        assert not self.model.is_xshards_distributed()
        # test predict
        yhat = self.model.predict(horizon=self.horizon)

        # test save load
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat_loaded = loaded_model.predict(horizon=self.horizon)
        yhat_id = yhat_loaded["id"]
        np.testing.assert_equal(yhat_id, self.id)
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_array_almost_equal(yhat, yhat_loaded, decimal=4)
        # test evaluate
        target_value = dict({"y": self.data_new})
        assert self.model.evaluate(target_value=target_value, metric=['mse'])
        # test fit_incremental
        self.model.fit_incremental({'y': self.data_new})  # 1st time
        self.model.fit_incremental({'y': self.data_new})  # 2nd time
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr["prediction"]
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)

    def test_forecast_ndarray_error(self):
        # is_xshards_distributed
        with self.assertRaises(Exception) as context:
            self.model.is_xshards_distributed()
        self.assertTrue('You should run fit before calling is_xshards_distributed()'
                        in str(context.exception))

        # fit
        input = dict({'data': self.data})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))

        input = dict({'y': "abc"})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue("the value of y should be an ndarray" in str(context.exception))

        id_diff = np.arange(200)
        input = dict({'id': id_diff, 'y': self.data})
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue("the length of the id array should be equal to the number of"
                        in str(context.exception))

        input = dict({'id': self.id, 'y': self.data})
        self.model.fit(input)
        with self.assertRaises(Exception) as context:
            self.model.fit(input)
        self.assertTrue('This model has already been fully trained' in str(context.exception))

        # fit_incremental
        data_id_diff = {'id': self.id - 1, 'y': self.data_new}
        with self.assertRaises(ValueError) as context:
            self.model.fit_incremental(data_id_diff)
        self.assertTrue('The input ids in fit_incremental differs from input ids in fit'
                        in str(context.exception))

        model1 = TCMFForecaster(y_iters=1, init_FX_epoch=1, max_FX_epoch=1, max_TCN_epoch=1,
                                alt_iters=2)
        model1.fit({'y': self.data})
        with self.assertRaises(ValueError) as context:
            model1.fit_incremental(data_id_diff)
        self.assertTrue('Got valid id in fit_incremental and invalid id in fit.'
                        in str(context.exception))

        # evaluate
        target_value_fake = dict({"data": self.data_new})
        with self.assertRaises(Exception) as context:
            self.model.evaluate(target_value=target_value_fake, metric=['mse'])
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))

    def test_forecast_tcmf_without_id(self):
        # construct data
        input = dict({'y': self.data})
        self.model.fit(input)
        assert not self.model.is_xshards_distributed()
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat = self.model.predict(horizon=self.horizon)
        yhat_loaded = loaded_model.predict(horizon=self.horizon)
        assert "id" not in yhat_loaded
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_array_almost_equal(yhat, yhat_loaded, decimal=4)

        target_value = dict({"y": self.data_new})
        self.model.evaluate(target_value=target_value, metric=['mse'])

        self.model.fit_incremental({'y': self.data_new})  # 1st time
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr["prediction"]
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)

        data_new_id = {'id': self.id, 'y': self.data_new}
        with self.assertRaises(ValueError) as context:
            self.model.fit_incremental(data_new_id)
        self.assertTrue('Got valid id in fit_incremental and invalid id in fit.'
                        in str(context.exception))

    def test_forecast_tcmf_xshards(self):
        from zoo.orca import OrcaContext
        import zoo.orca.data.pandas
        import pandas as pd
        OrcaContext.pandas_read_backend = "pandas"

        def preprocessing(df, id_name, y_name):
            id = df.index
            data = df.to_numpy()
            result = dict({id_name: id, y_name: data})
            return result

        def postprocessing(pred_results, output_dt_col_name):
            id_arr = pred_results["id"]
            pred_results = pred_results["prediction"]
            pred_results = np.concatenate((np.expand_dims(id_arr, axis=1), pred_results), axis=1)
            final_df = pd.DataFrame(pred_results, columns=["id"] + output_dt_col_name)
            final_df.id = final_df.id.astype("int")
            final_df = final_df.set_index("id")
            final_df.columns.name = "datetime"
            final_df = final_df.unstack().reset_index().rename({0: "prediction"}, axis=1)
            return final_df

        def get_pred(d):
            return d["prediction"]

        with tempfile.NamedTemporaryFile() as temp:
            data = np.random.rand(300, 480)
            df = pd.DataFrame(data)
            df.to_csv(temp.name)
            shard = zoo.orca.data.pandas.read_csv(temp.name)
        shard.cache()
        shard_train = shard.transform_shard(preprocessing, 'id', 'data')
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'cid', 'y')
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue("key `id` doesn't exist in x" in str(context.exception))
        with self.assertRaises(Exception) as context:
            self.model.is_xshards_distributed()
        self.assertTrue('You should run fit before calling is_xshards_distributed()'
                        in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'id', 'y')
        self.model.fit(shard_train)
        assert self.model.is_xshards_distributed()
        with self.assertRaises(Exception) as context:
            self.model.fit(shard_train)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            self.model.fit_incremental(shard_train)
        self.assertTrue('NotImplementedError' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname + "/model")
            loaded_model = TCMFForecaster.load(tempdirname + "/model", is_xshards_distributed=True)
        horizon = np.random.randint(1, 50)
        yhat_shard_origin = self.model.predict(horizon=horizon)
        yhat_list_origin = yhat_shard_origin.collect()
        yhat_list_origin = list(map(get_pred, yhat_list_origin))
        yhat_shard = loaded_model.predict(horizon=horizon)
        yhat_list = yhat_shard.collect()
        yhat_list = list(map(get_pred, yhat_list))
        yhat_origin = np.concatenate(yhat_list_origin)
        yhat = np.concatenate(yhat_list)
        assert yhat.shape == (300, horizon)
        np.testing.assert_equal(yhat, yhat_origin)
        output_dt_col_name = pd.date_range(start='2020-05-01', periods=horizon, freq='H').to_list()
        yhat_df_shards = yhat_shard.transform_shard(postprocessing, output_dt_col_name)
        final_df_list = yhat_df_shards.collect()
        final_df = pd.concat(final_df_list)
        final_df.sort_values("datetime", inplace=True)
        assert final_df.shape == (300 * horizon, 3)
        OrcaContext.pandas_read_backend = "spark"

    def test_forecast_tcmf_distributed(self):
        input = dict({'id': self.id, 'y': self.data})

        from zoo.orca import init_orca_context, stop_orca_context

        init_orca_context(cores=4, spark_log_level="INFO", init_ray_on_spark=True,
                          object_store_memory="1g")
        self.model.fit(input, num_workers=4)

        with tempfile.TemporaryDirectory() as tempdirname:
            self.model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, is_xshards_distributed=False)
        yhat = self.model.predict(horizon=self.horizon, num_workers=4)
        yhat_loaded = loaded_model.predict(horizon=self.horizon, num_workers=4)
        yhat_id = yhat_loaded["id"]
        np.testing.assert_equal(yhat_id, self.id)
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (self.num_samples, self.horizon)
        np.testing.assert_equal(yhat, yhat_loaded)

        self.model.fit_incremental({'y': self.data_new})
        yhat_incr = self.model.predict(horizon=self.horizon)
        yhat_incr = yhat_incr["prediction"]
        assert yhat_incr.shape == (self.num_samples, self.horizon)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, yhat, yhat_incr)

        target_value = dict({"y": self.data_new})
        assert self.model.evaluate(target_value=target_value, metric=['mse'])
        stop_orca_context()


if __name__ == "__main__":
    pytest.main([__file__])
