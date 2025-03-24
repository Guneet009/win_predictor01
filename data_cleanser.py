from sklearn.pipeline import Pipeline

import pandas as pd
from typing import List

class DataCleanser:
    def __init__(self,country:str,country_name:str,opp_list:str):
        self.country = country
        self.country_name = country_name
        self.opp_list = opp_list
    
    def get_data(self)->pd.DataFrame:
        df = pd.read_csv(f"D:\predictor01\data\\{self.country}_data.csv")
        return df
    
    def get_year(self,df:pd.DataFrame)->pd.DataFrame:
        df['year'] = df['Date'].str.extract(r'(\d{4})')
        df.drop(columns=['Date'],axis=1,inplace=True)
        return df
    
    def get_opp_country(self,df:pd.DataFrame)->pd.DataFrame:
        df['Opposition_Country'] = df['Series'].str.extract(rf'\b({self.opp_list})\b', expand=False)
        df.drop(columns=['Series'],axis=1,inplace=True)
        df.dropna(inplace=True)
        x = len(df)
        df['Country'] = [f'{self.country_name}' for i in range(x)]
        return df
    
    def get_result(self,df)->pd.DataFrame:
        df['Country_won'] = df['Result'].str.extract(r'\b(Australia|England|India|Afghanisthan|South Africa|Bangladesh|Pakistan|New Zealand)\b', expand=False)
        df.dropna(inplace=True)
        df['result'] = df['Country_won'].apply(lambda x: 1 if x == f'{self.country_name}' else 0)
        df.drop(columns=['Result','Country_won'],axis=1,inplace=True)
        return df

    