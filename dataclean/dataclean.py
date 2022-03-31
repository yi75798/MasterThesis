#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   dataclean.py
# Time    :   2022/03/30 10:46:50
# Author  :   Hsu, Liang-Yi 
# Email:   yi75798@gmail.com
# Description : Program of data clean for mater thesis.

import pandas as pd
import numpy as np
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from sklearn.impute import KNNImputer 
import os

# path = os.path.join(os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__))

## Rounding
def rounding(num, decimal=0):
    num = np.round(num, decimal)
    #num = float(num)
    return num

## KNNImputer
# # Find the optimal K.
# rmse = lambda y, yhat: np.sqrt(mean_squared_error(y, yhat))

# def optimize_k(data, target): 
#     errors = [] 
#     for k in range(1, 20, 2): 
#         imputer = KNNImputer(n_neighbors=k) 
#         imputed = imputer.fit_transform(data) 
#         df_imputed = pd.DataFrame(imputed, columns=data.columns) 
         
#         X = df_imputed.drop(target, axis=1) 
#         y = df_imputed[target] 
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
#         model = RandomForestRegressor() 
#         model.fit(X_train, y_train) 
#         preds = model.predict(X_test) 
#         error = rmse(y_test, preds) 
#         errors.append({'K': k, 'RMSE': error}) 
         
#     return errors


### Build up a data cleaner.
class Cleaner:
    def __init__(self, data_path: str, codebook_path: str):
        '''
        Build up a data cleaner.

        :data_path (str) : Path of the raw data. The raw data must be csv file.
        :codebook_path (str) : Path of the codebook file.
        '''
        self.df = pd.read_csv(data_path, encoding='utf_8_sig')
        self.codebook_var = pd.read_excel(codebook_path, sheet_name='VAR').set_index('var')
        self.codebook_value = pd.read_excel(codebook_path, sheet_name='VALUE').set_index('var')
    
    def data(self):
        '''
        Return the dataframe.

        :return: pd.DataFrame
        '''
        return self.df

    def codebook(self, var_value='VAR'):
        '''
        Return the codebook.

        :var_value (str): Optional. 
                          Return the codebook of var or values.
                          Default='VAR'

        :return: pd.DataFrame
        '''
        if var_value == 'VAR':
            return self.codebook_var
        elif var_value == 'VALUE':
            return self.codebook_value
    
    def select_var(self):
        '''
        Select Variables needed from raw data and rename columns.

        :return: class Cleaner
        '''
        var = list(self.codebook_var['origin_qno'])
        col_names = list(self.codebook_var.index)
        self.df = self.df[var]
        self.df.columns = col_names
        return self

    def defna(self):
        '''
        Define which values are NA.

        :return : class Cleaner
        '''
        var = list(self.codebook_var.index)

        for v in var:
            try:
                not_na = [int(i) for i in self.codebook_var['not_NA_value'].loc[v].split(',')]
                self.df[v] = np.where(~self.df[v].isin(not_na), np.nan, self.df[v])
            except AttributeError:
                continue
            except KeyError:
                return print('Error: Maybe you should call .select_var func first!')
                
        
        return self

    def recode(self):
        '''
        Recode old values to new values.

        :return: class Cleaner
        '''
        var = list(self.codebook_value.index)

        try:
            for v in var:
                old = [int(i) for i in self.codebook_value['old_value'].loc[v].split(',')]
                new = [int(i) for i in self.codebook_value['new_value'].loc[v].split(',')]
                replace_book = dict(zip(old, new))

                self.df[v].replace(replace_book, inplace=True)
        except KeyError:
                return print('Error: Maybe you should call .select_var func first!')
            
        return self
    
    def knn_fillna(self, k=10):
        '''
        Fill na by KNN.

        :k (int): Optional. The number of k. Default is 10.

        :return: class Cleaner
        '''
        print('Nums of na before filled:\n', self.df.isnull().sum())

        var = list(self.codebook_value.index)
        imputer = KNNImputer(n_neighbors=10) # default k=10. 
        imputed = imputer.fit_transform(self.df) 
        self.df = pd.DataFrame(imputed, columns=self.df.columns)

        print('======\nNums of na after filled:\n', self.df.isnull().sum())
        w = self.df['weight'] # keep the weight not to be apply rounding.
        self.df = self.df.drop('weight', axis=1).apply(rounding)
        self.df = pd.concat([self.df, w], axis=1)
        return self
    
    def clean_all(self, k=10):
        '''
        Clean data at once.

        :k (int): Optional. The number of k in KNN. Default is 10.

        :return class Cleaner
        '''
        self.select_var()
        self.defna()
        self.recode()
        self.knn_fillna()

        return self
    
    def output(self, outputpath: str):
        '''
        Output the cleaned dataframe into csv file.

        :outputpath (str): path to output.
        '''
        self.df.to_csv(outputpath, index=False, encoding='utf_8_sig')

    

if __name__ == '__main__':
    clean_data = Cleaner('testdata.csv', 'codebook_year.xlsx').clean_all()
    clean_data.output('testoutput.csv')
