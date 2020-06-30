import numpy as np
import pandas as pd
import pytest

from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
from zoo.zouwu.preprocessing.LastFill import LastFill

class TestDataImputation(ZooTestCase):
    
    def setup_method(self, method):
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()
    
    def teardown_method(self, method):
       pass
       
    def create_data(self):
        data = np.random.random_sample((5,5))
        mask = np.random.random_sample((5,5))
        mask[mask>=0.2] = 1
        mask[mask<0.2] = 0
        data[mask==0] = None
        df = pd.DataFrame(data)
        idx = pd.date_range(start='2020-07-01 00:00:00', end='2020-07-01 08:00:00', freq='2H')
        df.index = idx
        self.data = df
        
    def test_impute(self):
        last_fill = LastFill()
        imputed_data = last_fill.impute(self.data)
        assert imputed_data.isna().sum().sum()==0
    
    def test_evaluate(self):
        last_fill = LastFill()
        mse_missing = last_fill.evaluate(self.data, 0.1)
        imputed_data = last_fill.impute(self.data)
        mse = last_fill.evaluate(imputed_data, 0.1)
        
if __name__ == "__main__":
    pytest.main([__file__])     