# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import base_est

data = pd.read_csv('TRAIN.csv')
# top rows of the data
print(data.head())

data=data.dropna()

#Data Partition

X = data.drop('Churn Status', axis =1)
y = data['Churn Status']



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing',missing_values ="NaN")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer( remainder= 'passthrough',
    transformers=[('drop_columns', 'drop', ['Customer ID',
                                     'Most Loved Competitor network in in Month 1',
                                     'Most Loved Competitor network in in Month 2',
                                     'Network type subscription in Month 1',
                                     'Network type subscription in Month 2',
                                     'Total Call centre complaint calls',
                                     'Total Onnet spend ',
                                     'Total Spend in Months 1 and 2 of 2017' ]),
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


rf = Pipeline(steps=[ ('get_binary_columns',base_est.VariableEncoder1()),
                      ('preprocessor', preprocessor),
                      ('classifier', GradientBoostingClassifier())])

rf.fit(X,y)

#So i have to save model
import pickle
pickle.dump(rf, open('model.pkl','wb'))  

  
