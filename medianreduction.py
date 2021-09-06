from DataCleaning import import_pickle,export_pickle,import_selected,combine_df_province, check_province_voc,select_10,group_by_province
from anova_population import create_histogram, qqplot, group_stations, verify_vocs
from Classes import Station
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

'''This code uses data in the format of 'prov_data' -->


'''

def median_reduction(province,voc,path=None):
    '''
    Plots the changes in the 5-year medians for different stations over all the years for a given VOC.

    Inputs: province (int) --> Number code of the given province for the data
            voc (str) --> Name of the VOC
            path (str) --> Path to the folder you want to save the figure in

    Outputs: df_medians (Pandas DataFrame) --> DataFrame of the medians for each station over the years
    '''
    df_overall = pd.DataFrame(columns=['Year'])
    df_overall['Year']=[1989,1994,1999,2004,2009,2014]
    for df,station in prov_data[province]:
        meds = find_regular_medians(df,5,voc)
        df_overall[station]=[meds[year] for year in meds]
        urbanization=md['Urbanization'][md['NAPS ID']==station].iloc[0]
        prov = md['Province'][md['NAPS ID']==station].iloc[0]
        df_overall=df_overall.rename(columns={station:'{} ({})'.format(station,urbanization)})
    df_medians = df_overall.iloc[:,1:]
    fig,ax=plt.subplots()
    sns.lineplot(ax=ax,data=df_medians,markers=True)
    sns.set_style({'axes.facecolor': '#e8e8ea', 'axes.grid': True, 'font.family': 'Serif'})
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 12,
            }
    ax.set_xlabel('Year interval',fontdict=font)
    ax.set_ylabel('Median Concentration (ug/m3)',fontdict=font)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(['1989\n1994', '1994\n1999', '1999\n2004', '2004\n2009', '2009\n2014', '2014\n2020'],font)
    ax.set_ylim(bottom=0)
    ax.grid(b=True, which='major', axis='x', color='#d8d8df')
    plt.suptitle(
        '{} concentration over the years in {}'.format(voc,provinces[province]))
    if province==6:
        cols=3
    else:
        cols=2
    ax.legend(loc='upper right', title='Station', fancybox=False,ncol=cols,fontsize='x-small')
    plt.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
    return df_medians,df_overall

def median_reduction2(voc,path=None):
    df_overall = pd.DataFrame(columns=['Year'])
    years= [(1989, 1994), (1994, 1999), (1999, 2004),(2004,2009),(2009,2014),(2014,2020)]
    df_overall['Year']=['Province',1989,1994,1999,2004,2009,2014]
    for province in provinces:
        for df,station in prov_data[province]:
            meds = find_regular_medians(df,5,voc)
            prov = md['Province'][md['NAPS ID'] == station].iloc[0]
            station_col = [meds[year] for year in meds]
            station_col.insert(0,prov)
            df_overall[station]=station_col
            urbanization=md['Urbanization'][md['NAPS ID']==station].iloc[0]
            df_overall=df_overall.rename(columns={station:'{} ({})'.format(station,urbanization)})
    palette = sns.color_palette('Set2',10)
    # fig,ax=plt.subplots(figsize=(8,5))
    df_medians = df_overall.iloc[1:,:]
    #sns.lineplot(ax=ax,data=df_medians.astype(float))
    for station in df_overall.columns[1:]:
        prov_indices = {'NL':0,'NS':1,'NB':2,'QC':3,'ON':4,'MB':5,'SK':6,'AB':7,'BC':8,'NU':9}
        prov = df_overall[station].iloc[0]
        line=plt.plot(df_overall['Year'].iloc[1:].astype(float),df_overall[station].iloc[1:].astype(float),color=palette[prov_indices[prov]],linestyle='--', marker='o')
        #plt.grid({'axes.facecolor': '#e8e8ea', 'axes.grid': True, 'font.family': 'Serif'})
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 12,
                }
        # ax.set_xlabel('Year interval',fontdict=font)
        # ax.set_ylabel('Median Concentration (ug/m3)',fontdict=font)
        # ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
        # ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
        # ax.set_xticks(np.arange(6))
        # ax.set_xticklabels(['1989\n1994', '1994\n1999', '1999\n2004', '2004\n2009', '2009\n2014', '2014\n2020'],font)
        # ax.set_ylim(bottom=0)
        # ax.grid(b=True, which='major', axis='x', color='#d8d8df')
    plt.xlabel('Year interval',fontdict=font)
    plt.ylabel('Median Concentration (ug/m3)',fontdict=font)
    plt.yticks(np.arange(0,4,0.5))
    #ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    plt.xticks(np.arange(1989,2019,5),['1989\n1994', '1994\n1999', '1999\n2004', '2004\n2009', '2009\n2014', '2014\n2020'],fontfamily='serif')
    plt.ylim(bottom=0)
    plt.suptitle('{} concentration over the years in Canada'.format(voc),fontproperties=font)
    handles = [mpl.lines.Line2D([], [], color=palette[4], label='ON'),
               mpl.lines.Line2D([], [], color=palette[3], label='QC'),
               mpl.lines.Line2D([], [], color=palette[7], label='AB'),
               mpl.lines.Line2D([], [], color=palette[8], label='BC'),
               mpl.lines.Line2D([], [], color=palette[6], label='SK'),
               mpl.lines.Line2D([], [], color=palette[2], label='NB'),
               mpl.lines.Line2D([], [], color=palette[1], label='NS'),
               mpl.lines.Line2D([], [], color=palette[5], label='MB'),
               mpl.lines.Line2D([], [], color=palette[0], label='NL'),
               mpl.lines.Line2D([], [], color=palette[9], label='NU')]
    plt.legend(loc='upper right', title='Province', handles=handles,fancybox=False,ncol=2)
    plt.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
    return df_overall

'''Running zone'''
prov_data = import_pickle("E://Summer research/pythonProject/data/10_years_provinces.pickle")[0]

province = 6
voc = 'Butane'
provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
md = pd.read_excel("E:\VOC Project\Random\StationsMetadata.xlsx")
# a,b=median_reduction(6)
# df_o=median_reduction2('Dichloromethane')
# for province in provinces:
#     median_reduction(province,voc,"E://Summer research/pythonProject/figures/Median reduction/{}/{}.png".format(voc,provinces[province]))
# median_reduction2('Styrene')