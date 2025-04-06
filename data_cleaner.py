import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
TARGET_COLUMN = 'winner'
# Custom transformer to combine columns
class CombineCategoricals(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Combine all columns into a single DataFrame for consistent encoding
        return X.astype(str)



def save_object(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)
        
    with open(file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)


class DataCleaner:
    def __init__(self,df:pd.DataFrame):
        self.df = df
    
    def get_dtypes(self)->List:
        obj_col = [col for col in self.df.columns if self.df[col].dtype == 'O']
        num_col = [col for col in self.df.columns if self.df[col].dtype != 'O']
        return obj_col,num_col
    
    
    def create_class(self,obj_col,num_col):
        num_pipeline = Pipeline(
            steps=[
                ("scaler",StandardScaler())
            ]
        )
        
        cat_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("combine", CombineCategoricals()),
                ("ohe",OneHotEncoder())        
            ]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline",num_pipeline,num_col)
                ("cat_pipeline",cat_pipeline,obj_col)
            ]
        )
        return preprocessor
    
    def initiate_transformation(self,train_data:pd.DataFrame,test_data:pd.DataFrame):
        # input_feature_train = train_data.drop(columns=[TARGET_COLUMN],axis=1)
        # output_feature_train = train_data[TARGET_COLUMN]
        # input_feature_test = test_data.drop(columns=[TARGET_COLUMN],axis=1)
        # output_feature_test = test_data[TARGET_COLUMN]
        cat_col,num_col = self.get_dtypes()
        preprocessor = self.create_class(obj_col=cat_col,num_col=num_col)
        train_arr = preprocessor.fit_transform(train_data)
        test_arr = preprocessor.transform(test_data)
        save_object(file_path="Artifacts/preprocessor.pkl",obj=preprocessor)
        
        return train_arr,test_arr
        
        
                
    
