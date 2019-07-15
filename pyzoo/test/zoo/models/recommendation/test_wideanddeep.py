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
        sparkConf = init_spark_conf().setMaster("local[4]") \
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
        input = JTensor.sparse(np.array([1, 3, 5, 2, 4, 6]),
                               np.array([[0, 0, 0, 0, 0, 0], [1, 2, 5, 8, 90, 100]]),
                               np.array([2, 124]))
        model = WideAndDeep(5, self.column_info, "wide")
        output = model.forward(input)

    def test_deep_indicator_forward_backward(self):
        column_info = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3])
        model = WideAndDeep(5, column_info, "deep")
        input = np.random.randint(2, size=(2, 24))
        self.assert_forward_backward(model, input)

    def test_deep_embedding_forward_backward(self):
        column_info = ColumnFeatureInfo(
            embed_cols=["userId", "itemId"],
            embed_in_dims=[100, 100],
            embed_out_dims=[20, 20])
        model = WideAndDeep(5, column_info, "deep")
        model.summary()
        input = np.random.randint(1, 100, size=(10, 2))
        self.assert_forward_backward(model, input)

    def test_deep_continous_forward_backward(self):
        column_info = ColumnFeatureInfo(
            continuous_cols=["age", "whatever"])
        model = WideAndDeep(5, column_info, "deep")
        model.summary()
        input = np.random.randint(1, 100, size=(100, 2))
        self.assert_forward_backward(model, input)

    def test_save_load(self):
        column_info = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3])
        model = WideAndDeep(5, column_info, "deep")
        input_data = np.random.randint(2, size=(2, 24))
        self.assert_zoo_model_save_load(model, input_data)

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

    def test_compile_fit(self):
        column_info = self.column_info
        model = WideAndDeep(5, column_info, "wide_n_deep")
        data = self.data_in.rdd.map(lambda row: to_user_item_feature(row, column_info)) \
            .map(lambda user_item_feature: user_item_feature.sample)
        model.compile(loss=SparseCategoricalCrossEntropy(zero_based_label=False),
                      optimizer="adam",
                      metrics=["mae"])
        model.fit(data, nb_epoch=1)

    def test_deep_merge(self):
        def get_input(column_info):
            input_ind = Input(shape=(sum(column_info.indicator_dims),))
            input_emb = Input(shape=(len(column_info.embed_in_dims),))
            input_con = Input(shape=(len(column_info.continuous_cols),))
            return(input_ind, input_emb, input_con)

        column_info1 = self.column_info
        model1 = WideAndDeep(5, column_info1, "deep")
        (input_ind1, input_emb1, input_con1) = get_input(column_info1)
        (input1, merged_list1) = model1._deep_merge(input_ind1, input_emb1, input_con1)
        assert len(input1) == 3
        assert len(merged_list1) == 4

        column_info2 = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3],
            embed_cols=["userId", "itemId"],
            embed_in_dims=[100, 100],
            embed_out_dims=[20, 20])
        model2 = WideAndDeep(5, column_info2, "deep")
        (input_ind2, input_emb2, input_con2) = get_input(column_info2)
        (input2, merged_list2) = model2._deep_merge(input_ind2, input_emb2, input_con2)
        assert len(input2) == 2
        assert len(merged_list2) == 3

        column_info3 = ColumnFeatureInfo(
            indicator_cols=["occupation", "gender"],
            indicator_dims=[21, 3],
            continuous_cols=["age"])
        model3 = WideAndDeep(5, column_info3, "deep")
        (input_ind3, input_emb3, input_con3) = get_input(column_info3)
        (input3, merged_list3) = model3._deep_merge(input_ind3, input_emb3, input_con3)
        assert len(input3) == 2
        assert len(merged_list3) == 2


if __name__ == "__main__":
    pytest.main([__file__])
