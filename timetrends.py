from DataCleaning import import_pickle,export_pickle,import_selected,combine_df_province, check_province_voc,select_10,group_by_province
from anova_population import create_histogram, qqplot, group_stations, verify_vocs
from Classes import Station
from boxplots import time_trend_singular
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
import seaborn as sns
import pylab as py
import math
import sklearn
from scatter import find_regular_medians

#get the collapsed data
data = import_pickle("E://Summer research/pythonProject/data/colldatav4.pickle")[0]

#make a Stations object
# station1 = Station("S100111",data)

#get the df for Ethane in 1989-1999
# s1_ethane = station1.get_voc('Dichlorobenzene',1989,1999)

def plot_time_series(df,voc):
    '''Takes in a station df & plots a time series'''
    #drop duplicates of time
    df.drop_duplicates(subset=['Compounds'],keep='first',inplace=True)
    time = df.iloc[:,0]
    voc = df.iloc[:,1]
    plt.plot(time,voc)
    plt.show()


# df = data['S100111']

'''Number of data points > 700'''
stations={'Alberta':['S090121','S090227','S090130'],'British Columbia':['S100111', 'S100133', 'S100137', 'S100119',
'S102001', 'S100202'],'Quebec':['S050103', 'S050104', 'S050115', 'S050121', 'S054102', 'S054401', 'S054501', 'S055201'
, 'S050129'],'Ontario':['S060101', 'S060418', 'S060512', 'S061004', 'S063201', 'S060211', 'S063601', 'S060104',
'S060413', 'S062601', 'S064401', 'S060403', 'S064601', 'S060903', 'S060428'],'Manitoba':['S070119'],'Saskatchewan':
['S080110'],'Nova Scotia':['S030118','S030501'],'New Brunswick':['S040203','S040501','S040208']}

# province = 'Alberta'
# for station in stations[province]:
#     time_trend_singular(alldata=data,station=station,voc='Styrene',ylim=2,window=12,path='E://Summer research/pythonProject/figures/Time Series/Styrene/{}/{}.png'.format(province,station))

time_trend_singular(alldata=data,station='S090227',voc='Dichloromethane',ylim=1.5,window=12)