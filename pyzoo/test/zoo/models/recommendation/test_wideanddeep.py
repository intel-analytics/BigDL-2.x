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

from pyspark.sql.functions import col, udf
from zoo.models.recommendation import *
from zoo.models.recommendation.utils import *
from zoo.common.nncontext import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase

np.random.seed(1337)  # for reproducibility


class TestWideAndDeep(ZooTestCase):

    def setup_method(self, method):
        sparkConf = init_spark_conf().setMaster("local[4]")\
            .setAppName("test wide and deep")
        self.sc = init_nncontext(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        data_path = os.path.join(os.path.split(__file__)[0], "../../resources/recommender")
        categorical_gender_udf = udf(lambda gender:
                                     categorical_from_vocab_list(gender, ["F", "M"], start=1))
        bucket_udf = udf(lambda feature1, feature2:
                         hash_bucket(str(feature1) + "_" + str(feature2), bucket_size=100))
        self.data_in = self.sqlContext.read.parquet(data_path) \
            .withColumn("gender", categorical_gender_udf(col("gender")).cast("int")) \
            .withColumn("occupation-gender",
                        bucket_udf(col("occupation"), col("gender")).cast("int"))
        self.column_info = ColumnFeatureInfo(
            wide_base_cols=["occupation", "gender"],
            wide_base_dims=[21, 3],
            wide_cross_cols=["occupation-gender"],
            wide_cross_dims=[100],
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3],
            embed_cols=["userId", "itemId"],
            embed_in_dims=[100, 100],
            embed_out_dims=[20, 20],
            continuous_cols=["age"])

    def test_wide_forward_backward(self):
        column_info = ColumnFeatureInfo(
            wide_base_cols=["occupation", "gender"],
            wide_base_dims=[21, 3],
            wide_cross_cols=["occupation-gender"],
            wide_cross_dims=[100])
        model = WideAndDeep(5, column_info, "wide")
        data = self.data_in.rdd.map(lambda r: get_wide_tensor(r, column_info))
        data.map(lambda input_data: self.assert_forward_backward(model, input_data))

    def test_deep_forward_backward(self):
        column_info = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3])
        model = WideAndDeep(5, column_info, "deep")
        data = self.data_in.rdd.map(lambda r: get_deep_tensor(r, column_info))
        data.map(lambda input_data: self.assert_forward_backward(model, input_data))

    def test_wide_and_deep_forward_backward(self):
        column_info = self.column_info
        model = WideAndDeep(5, column_info, "wide_n_deep")
        data = self.data_in.rdd.map(lambda r: [get_wide_tensor(r, column_info),
                                               get_deep_tensor(r, column_info)])
        data.map(lambda input_data: self.assert_forward_backward(model, input_data))

    def test_save_load(self):
        column_info = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3])
        model = WideAndDeep(5, column_info, "deep")
        input_data = get_deep_tensor(self.data_in.take(1)[0], column_info)
        self.assert_zoo_model_save_load(model, input_data.reshape((1, input_data.shape[0])))
        print(self.column_info)

    def test_predict_recommend(self):
        column_info = self.column_info
        model = WideAndDeep(5, column_info, "wide_n_deep")
        data = self.data_in.rdd.map(lambda row: to_user_item_feature(row, column_info))
        predictions = model.predict_user_item_pair(data)
        print(predictions.take(1)[0])
        recommended_items = model.recommend_for_user(data, max_items=3)
        print(recommended_items.take(1)[0])
        recommended_users = model.recommend_for_item(data, max_users=4)
        print(recommended_users.take(1)[0])

    def test_negative_sample(self):
        negative_df = get_negative_samples(self.data_in)


if __name__ == "__main__":
    pytest.main([__file__])
