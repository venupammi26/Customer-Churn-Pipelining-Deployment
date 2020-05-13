# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
import numpy as np

class VariableEncoder1(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        x_dataset['MostLovedCompetitornetworkininMonth2_Uxaa'] = (x_dataset['Most Loved Competitor network in in Month 2'] == 'Uxaa')*1
        
        
        for i in x_dataset.select_dtypes(include=['int64', 'float64']).columns:
            x_dataset[i].fillna(x_dataset[i].median(),inplace= True)
         
        for i in x_dataset.select_dtypes(include=['object']).columns:
            x_dataset[i].fillna(str(x_dataset[i].mode()),inplace= True)
            
        
        return x_dataset
