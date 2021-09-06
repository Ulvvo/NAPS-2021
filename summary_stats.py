from DataCleaning import import_pickle,export_pickle
from anova_population import create_histogram, qqplot
from Classes import Station
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
import seaborn as sns
import pylab as py

alldata = import_pickle('E://Summer research/pythonProject/data/colldatav3.pickle')[0]
allvocs = import_pickle('E://Summer research/pythonProject/data/allvocs.pickle')[0]

def summarize_means():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].mean(axis=0,skipna=True)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

def summarize_medians():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].median(axis=0,skipna=True)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

'''Doesn't work for some reason'''
def summarize_modes():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].mode(axis=0)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

def summarize_stds():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].std(axis=0,skipna=True)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

def summarize_skew():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].skew(axis=0)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

def summarize_kurtosis():
    '''Summarize the mean concentrations of all the VOCs for all the stations'''
    df_dict = {'Station':[]}
    for voc in allvocs:
        df_dict[voc]=[] #create a column for every VOC
    df_final = pd.DataFrame(df_dict)
    for station in alldata:
        df = alldata[station]
        all_means = df.iloc[:,1:].kurt(axis=0)
        new_df_dict = {VOC:all_means[VOC] for VOC in all_means.axes[0].tolist()}
        new_df = pd.DataFrame(new_df_dict,index=[0])
        new_df.insert(0,'Station',station)
        df_final=pd.concat([df_final,new_df],axis=0,ignore_index=True,sort=False)
    return df_final

# df_kurt = summarize_kurtosis()
df = alldata['S010102']
arr = df['Ethane']
newarr = arr.dropna(axis=0)
ans = scipy.stats.lognorm.fit(newarr)
sns.displot(arr,bins=range(30),kde=True,color='#607c8e')
sns.kdeplot(scipy.stats.lognorm.pdf())