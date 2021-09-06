import pandas as pd
import numpy as np
from DataCleaning import get_paths, get_paths2, skiprows,create_dfs, create_dfs2, create_dfs3
import pickle


'''(USE RARELY) Removing columns of data that are made up of a majority NaN'''
def remove_cols(df):
    '''
    Takes in a DataFrame and removes columns that have a large proportion (>50%) of NaNs in any of its columns
    input: df (df) --> eg. for station 1, year 2010
                VOC1    VOC2    ...
    1/1/2010    1.346   2.345   ...
    7/1/2010    3.4l5   NaN     ...
    13/1/2010   0.746   1.957   ...
    ...         ...     ...     ...

    outputs: df with cols containing >50% Nans removed
             na_sum (dict) --> {VOC1: number of NaNs,...} eg. {'Ethane':0, 'Ethylene':5,...}
    '''
    na_sum = {} #create an empty dictionary
    labels = [] #empty list for column names with more than half empty columns
    for col in df.columns:
        na = df[col].isnull().sum() #count NaNs in column
        na_sum[col] = na #add number to dictionary
        if na >= 0.5*len(df):
            labels.append(col) #add column name to labels if more than half the column in NaNs
    df.drop(axis=1,labels=labels, inplace = True) #drop the columns in labels list
    return df, na_sum

