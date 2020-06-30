from abc import ABC, abstractmethod


class BaseImpute(ABC):
    """
    base model for data imputation
    """
    
    @abstractmethod
    def impute(self, df):
        """
        fill in missing values in the dataframe
        :param df: dataframe containing missing values
        :return: dataframe without missing values
        """
        raise NotImplementError
        
    @abstractmethod
    def evaluate(self, df, drop_rate):
        """
        randomly drop some values and evaluate the data imputation method
        :param df: input dataframe (better without missing values)
        :param drop_rate: percentage value of randomly dropping data
        :return: MSE results
        """
        raise NotImplementError