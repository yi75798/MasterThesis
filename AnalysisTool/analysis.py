#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   analysis.py
# Time    :   2022/03/31 15:00:54
# Author  :   Hsu, Liang-Yi 
# Email:   yi75798@gmail.com
# Description : Analysis tool for master thesis. Include frequency table, cross table,
#               regression model and plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

## Rounding
def rounding(num, decimal=0):
    num = np.round(num, decimal)
    #num = float(num)
    return num


class Table:
    def __init__(self, data):
        '''
        :data (pandas.Dataframe)
        '''
        self.df = data
    
    def freq(self, var:str, w=None, label=None):
        '''
        Frequency distribution table.

        :var (str): The variable want to inspect frequency distribution.
        :w (str): Optional.
                  The column which will be used to weighted by.
                  Defult=None.
        :label (lsit): Optional.
                       The label of the values
        
        :return : pd.DataFrame
        '''
        df = self.df
        if w:
            a = pd.Series(df[[var, w]].groupby(var).sum()[w]) / df[w].sum()
            if label:
                b = label
            else:
                b = a.index
            c = np.round(a.values, 2)
            d = rounding(df[[var, w]].groupby(var).sum()[w])
            df_temp = pd.DataFrame({'Label': b, 'Num': d, 'Freq': c})
            return df_temp
        else:
            df[w] = 1
            a = pd.Series(df[[var, w]].groupby(var).sum()[w]) / df[w].sum()
            if label:
                b = label
            else:
                b = a.index
            c = np.round(a.values, 2)
            d = rounding(df[[var, w]].groupby(var).sum()[w])
            df_temp = pd.DataFrame({'Label': b, 'Num': d, 'Freq': c})
            return df_temp
    
    def cross(self, r_var: str, c_var: str, w= None, percent_by='row'):
        '''
        Cross Table.

        : r_var (str): The variable of row.
        : c_var (str): The variable of column.

        : return : pd.DataFrame
        '''
        df = self.df
        if w:
            if percent_by == 'row':
                df_num = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                    margins=True).round(0).astype(int)
                df_freq = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                     margins=True,
                                     normalize='index').round(2)
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_nums', '_rfreq'])
                return df_cross
            elif percent_by== 'col':
                df_num = pd.crosstab(df[r_var], df[c_var], values=df[w], aggfunc=sum,
                                     margins=True).round(0).astype(int)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                     margins=True,
                                     normalize='columns').round(2)
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_nums', '_cfreq'])
                return df_cross
        else:
            if percent_by == 'row':
                df_num = pd.crosstab(df[r_var], df[c_var],
                                     margins=True)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                     margins=True,
                                     normalize='index').round(2)
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_nums', '_rfreq'])
                return df_cross
            elif percent_by== 'col':
                df_num = pd.crosstab(df[r_var], df[c_var],
                                     margins=True)
                df_freq = pd.crosstab(df[r_var], df[c_var],
                                     margins=True,
                                     normalize='columns').round(2)
                df_cross = pd.merge(df_num, df_freq, on=r_var, suffixes=['_nums', '_cfreq'])
                return df_cross

if __name__ == '__main__':
    data = pd.read_csv('test_data.csv')
    df = Table(data)
    freq = df.freq('d_sat', label=['非常不滿意', '不滿意', '滿意', '非常滿意'])
    display(freq)

    cross = df.cross('d_sat', 'd_sup', percent_by='row', w='weight')
    display(cross)
    
    




