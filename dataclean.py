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
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error 
from sklearn.impute import KNNImputer 
import os

# path = os.path.join(os.path.dirname(__file__))
# os.chdir(os.path.dirname(__file__))

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

        :return: pd.DataFrame
        '''
        var = list(self.codebook_var['origin_qno'])
        col_names = list(self.codebook_var.index)
        df_selected = self.df[var]
        df_selected.columns = col_names
        return df_selected

    def defna(self):
        '''
        Define which values are NA.

        :return : pd.DataFrame
        '''
        df = self.select_var()
        var = list(self.codebook_var.index)

        for v in var:
            try:
                not_na = [int(i) for i in self.codebook_var['not_NA_value'].loc[v].split(',')]
                df[v] = np.where(~df[v].isin(not_na), np.nan, df[v])
            except AttributeError:
                continue
        
        return df

    def recode(self):
        '''
        Recode old values to new values.

        :return: pd.DataFrame
        '''
        df = self.select_var()
        var = list(self.codebook_value.index)

        for v in var:
            old = [int(i) for i in self.codebook_value['old_value'].loc[v].split(',')]
            new = [int(i) for i in self.codebook_value['new_value'].loc[v].split(',')]
            replace_book = dict(zip(old, new))

            df[v].replace(replace_book, inplace=True)
        
        return df

if __name__ == '__main__':
    df = Cleaner('TEDS2008.csv', 'codebook_2008.xlsx')
    df_data = df.data()
    cb_var = df.codebook()
    cd_val = df.codebook('VALUE')

    selected = df.select_var()
    na = df.defna()

    recode = df.recode()


