import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from zoo.zouwu.preprocessing.abstract import BaseImpute

class LastFill(BaseImpute):
    """
    Impute missing data with last seen value
    """
    def __init__(self):
        """
        Construct model for last filling method
        """
        pass
        
    def impute(self, df):
        """
        impute data
        :params df: input dataframe
        :return: imputed dataframe
        """
        df.iloc[0] = df.iloc[0].fillna(0)
        return df.fillna(method='pad')
        
    def evaluate(self, df, drop_rate):
        """
        evaluate model by randomly drop some value
        :params df: input dataframe
        :params drop_rate: percentage value that will be randomly dropped
        :return: MSE results
        """
        missing = df.isna()*1
        missing = missing.to_numpy()
        mask = np.zeros(df.shape[0]*df.shape[1])
        idx = np.random.choice(mask.shape[0], int(mask.shape[0]*drop_rate), replace = False)
        mask[idx] = 1
        mask = np.reshape(mask, (df.shape[0], df.shape[1]))
        np_df = df.to_numpy()
        np_df[mask==1] = None
        new_df = pd.DataFrame(np_df)
        impute_df = self.impute(new_df)
        pred = impute_df.to_numpy()
        true = df.to_numpy()
        pred[missing==1] = 0
        true[missing==1] = 0
        return [metrics.mean_squared_error(true[:,0], pred[:,0]),metrics.mean_squared_error(true[:,1], pred[:,1])]