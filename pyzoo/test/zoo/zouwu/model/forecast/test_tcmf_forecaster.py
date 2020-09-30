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

    def test_forecast_tcmf(self):
        model = TCMFForecaster(y_iters=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)
        horizon = np.random.randint(1, 50)
        # construct data
        id = np.arange(300)
        data = np.random.rand(300, 480)
        input = dict({'data': data})
        with self.assertRaises(Exception) as context:
            model.fit(input)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))
        input = dict({'id': id, 'y': data})
        with self.assertRaises(Exception) as context:
            model.is_distributed()
        self.assertTrue('You should run fit before calling is_distributed()'
                        in str(context.exception))
        model.fit(input)
        assert not model.is_distributed()
        with self.assertRaises(Exception) as context:
            model.fit(input)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            model.fit(input, incremental=True)
        self.assertTrue('NotImplementedError' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, distributed=False)
        yhat = model.predict(x=None, horizon=horizon)
        yhat_loaded = loaded_model.predict(x=None, horizon=horizon)
        yhat_id = yhat_loaded["id"]
        assert (yhat_id == id).all()
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (300, horizon)
        assert (yhat == yhat_loaded).all()
        target_value = np.random.rand(300, horizon)
        target_value = dict({"y": target_value})
        assert model.evaluate(x=None, target_value=target_value, metric=['mse'])
         # inject new data
        x_new = np.random.rand(300, horizon)
        model.fit_incremental({'y': x_new})  # 1st time
        # model.fit(x_new, incremental=True)  # 2nd time
        yhat = model.predict(x=None, horizon=horizon)
        yhat = yhat["prediction"]
        assert yhat.shape == (300, horizon)

    def test_forecast_tcmf_without_id(self):
        model = TCMFForecaster(y_iters=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)
        horizon = np.random.randint(1, 50)
        # construct data
        id = np.arange(200)
        data = np.random.rand(300, 480)
        input = dict({'y': "abc"})
        with self.assertRaises(Exception) as context:
            model.fit(input)
        self.assertTrue("the value of y should be an ndarray" in str(context.exception))
        input = dict({'id': id, 'y': data})
        with self.assertRaises(Exception) as context:
            model.fit(input)
        self.assertTrue("the length of the id array should be equal to the number of"
                        in str(context.exception))
        input = dict({'y': data})
        model.fit(input)
        assert not model.is_distributed()
        with self.assertRaises(Exception) as context:
            model.fit(input)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, distributed=False)
        yhat = model.predict(x=None, horizon=horizon)
        yhat_loaded = loaded_model.predict(x=None, horizon=horizon)
        assert "id" not in yhat_loaded
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (300, horizon)
        assert (yhat == yhat_loaded).all()
        target_value = np.random.rand(300, horizon)
        target_value_fake = dict({"data": target_value})
        with self.assertRaises(Exception) as context:
            model.evaluate(x=None, target_value=target_value_fake, metric=['mse'])
        self.assertTrue("key y doesn't exist in y" in str(context.exception))
        target_value = dict({"y": target_value})
        model.evaluate(x=None, target_value=target_value, metric=['mse'])

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

        model = TCMFForecaster(y_iters=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)

        with tempfile.NamedTemporaryFile() as temp:
            data = np.random.rand(300, 480)
            df = pd.DataFrame(data)
            df.to_csv(temp.name)
            shard = zoo.orca.data.pandas.read_csv(temp.name)
        shard.cache()
        shard_train = shard.transform_shard(preprocessing, 'id', 'data')
        with self.assertRaises(Exception) as context:
            model.fit(shard_train)
        self.assertTrue("key `y` doesn't exist in x" in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'cid', 'y')
        with self.assertRaises(Exception) as context:
            model.fit(shard_train)
        self.assertTrue("key `id` doesn't exist in x" in str(context.exception))
        with self.assertRaises(Exception) as context:
            model.is_distributed()
        self.assertTrue('You should run fit before calling is_distributed()'
                        in str(context.exception))
        shard_train = shard.transform_shard(preprocessing, 'id', 'y')
        model.fit(shard_train)
        assert model.is_distributed()
        with self.assertRaises(Exception) as context:
            model.fit(shard_train)
        self.assertTrue('This model has already been fully trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            model.fit(shard_train, incremental=True)
        self.assertTrue('NotImplementedError' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname + "/model")
            loaded_model = TCMFForecaster.load(tempdirname + "/model", distributed=True)
        horizon = np.random.randint(1, 50)
        yhat_shard_origin = model.predict(x=None, horizon=horizon)
        yhat_list_origin = yhat_shard_origin.collect()
        yhat_list_origin = list(map(get_pred, yhat_list_origin))
        yhat_shard = loaded_model.predict(x=None, horizon=horizon)
        yhat_list = yhat_shard.collect()
        yhat_list = list(map(get_pred, yhat_list))
        yhat_origin = np.concatenate(yhat_list_origin)
        yhat = np.concatenate(yhat_list)
        assert yhat.shape == (300, horizon)
        assert (yhat == yhat_origin).all()
        output_dt_col_name = pd.date_range(start='2020-05-01', periods=horizon, freq='H').to_list()
        yhat_df_shards = yhat_shard.transform_shard(postprocessing, output_dt_col_name)
        final_df_list = yhat_df_shards.collect()
        final_df = pd.concat(final_df_list)
        final_df.sort_values("datetime", inplace=True)
        assert final_df.shape == (300 * horizon, 3)
        OrcaContext.pandas_read_backend = "spark"

    def test_forecast_tcmf_distributed(self):
        model = TCMFForecaster(y_iters=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)
        horizon = np.random.randint(1, 50)
        # construct data
        id = np.arange(300)
        data = np.random.rand(300, 480)
        input = dict({'id': id, 'y': data})

        from zoo.orca import init_orca_context, stop_orca_context

        init_orca_context(cores=4, spark_log_level="INFO", init_ray_on_spark=True,
                          object_store_memory="1g")
        model.fit(input, num_workers=4)

        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, distributed=False)
        yhat = model.predict(x=None, horizon=horizon, num_workers=4)
        yhat_loaded = loaded_model.predict(x=None, horizon=horizon, num_workers=4)
        yhat_id = yhat_loaded["id"]
        assert (yhat_id == id).all()
        yhat = yhat["prediction"]
        yhat_loaded = yhat_loaded["prediction"]
        assert yhat.shape == (300, horizon)
        np.testing.assert_equal(yhat, yhat_loaded)
        target_value = np.random.rand(300, horizon)
        target_value = dict({"y": target_value})
        assert model.evaluate(x=None, target_value=target_value, metric=['mse'])
        stop_orca_context()


if __name__ == "__main__":
    pytest.main([__file__])
