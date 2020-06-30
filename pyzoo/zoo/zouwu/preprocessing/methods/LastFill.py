import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from zoo.zouwu.preprocessing.methods.abstract import BaseModel

class LastFill(BaseModel):
    def __init__(self):
        """
        Construct model for last filling method
        """
        pass
        
    def imputation(self, df):
        df.iloc[0].fillna(0, replace=True)
        return df.fillna(method='pad')
        
    def train(self):
        pass
        
    def model_evaluation(self, df, drop_rate):
        mask = np.zeros(df.shape[0]*df.shape[1])
        idx = np.random.choice(mask.shape[0], int(mask.shape[0]*drop_rate), replace = False)
        mask[idx] = 1
        mask = np.reshape(mask, (df.shape[0], df.shape[1]))
        np_df = df.to_numpy()
        np_df[mask==1] = None
        new_df = pd.DataFrame(np_df)
        impute_df = self.imputation(new_df)
        pred = impute_df.to_numpy()
        true = df.to_numpy()
        return [metrics.mean_squared_error(true[:,0], pred[:,0]),metrics.mean_squared_error(true[:,1], pred[:,1])]