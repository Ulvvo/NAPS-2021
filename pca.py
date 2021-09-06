from DataCleaning import import_pickle,export_pickle,import_selected,combine_df_province, check_province_voc,select_10,group_by_province
from anova_population import create_histogram, qqplot, group_stations, verify_vocs
from Classes import Station
from boxplots import time_trend_singular,time_trends
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns
import pylab as py
import math
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scatter import find_regular_medians

data = import_pickle("E://Summer research/pythonProject/data/colldatav4.pickle")[0]
vocs_md = pd.read_excel("E:\Summer research\pythonProject\data\VOCs2.xlsx")
station = 'S100111'
year1 = pd.Timestamp(1989,1,1)
year2 = pd.Timestamp(2020,1,1)

def pca_clustering(data,station,year1,year2):
    df = data[station]
    df_chosen = df[(year1<=df['Compounds'])&(df['Compounds']<year2)].dropna(axis=1,how='all')
    df_new = df_chosen.iloc[:,1:].dropna(axis=0,how='all')
    all_vocs = df_new.columns.to_list()
    selected_vocs = [voc for voc in all_vocs if (df_new[voc].isna().sum()<=0.05*len(df_new)) & (len(df_new[df_new[voc]!=0])>=0.75*len(df_new))]
    #selected_vocs = [voc for voc in all_vocs if (df_chosen[voc].isna().sum()<=0.1*len(df_chosen))]
    # selected_vocs.append('Compounds')
    df_new = df_new[selected_vocs]
    df_new = df_new.dropna(axis=0,how='any')
    x = df_new
    # x = xlog.replace(to_replace=-np.inf,value=0)
    # df_standardized = StandardScaler().fit_transform(X=x)
    x = sklearn.preprocessing.power_transform(X=x,standardize=True)
    '''VOC sources: vehicles, industry, VCPs, Natural, Other '''
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X=x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])

    '''CLustering'''
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(principalDf)
    y_kmeans = kmeans.predict(principalDf)
    plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('PCA for different VOCs, station {}, years {}-{}'.format(station,year1.year,year2.year))
    plt.show()
    components = pd.DataFrame(pca.components_,columns=np.array(selected_vocs)).rename(index={0:'PC1',1:'PC2'}).transpose()
    return components

def pca_clustering2(data,station,year1,year2,path=None):
    df = data[station]
    df_chosen = df[(year1<=df['Compounds'])&(df['Compounds']<year2)].dropna(axis=1,how='all')
    df_new = df_chosen.iloc[:,1:].dropna(axis=0,how='all')
    all_vocs = df_new.columns.to_list()
    selected_vocs = [voc for voc in all_vocs if (df_new[voc].isna().sum()<=0.05*len(df_new)) & (len(df_new[df_new[voc]!=0])>=0.75*len(df_new))]
    #selected_vocs = [voc for voc in all_vocs if (df_chosen[voc].isna().sum()<=0.1*len(df_chosen))]
    # selected_vocs.append('Compounds')
    df_new = df_new[selected_vocs]
    df_new = df_new.dropna(axis=0,how='any')
    x = df_new
    # x = xlog.replace(to_replace=-np.inf,value=0)
    # df_standardized = StandardScaler().fit_transform(X=x)
    x = sklearn.preprocessing.power_transform(X=x,standardize=True)
    '''VOC sources: vehicles, industry, VCPs, Natural, Other '''
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X=x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    components = pd.DataFrame(pca.components_, columns=np.array(selected_vocs)).rename(
        index={0: 'PC1', 1: 'PC2'}).transpose()
    vocs = components.index.to_numpy()
    components.insert(0,'VOC',vocs)
    types = vocs_md['Type'][vocs_md['Name'].isin(vocs)].to_numpy()
    components.insert(1,'Type',types)
    n = vocs_md['N'][vocs_md['Name'].isin(vocs)].to_numpy()
    components.insert(2, 'N', n)
    fig,ax=plt.subplots()
    sns.scatterplot(ax=ax,x='PC1',y='PC2',data=components,hue='N',style='Type',s=80)
    ax.set_title('PCA for station {}'.format(station))
    handles, labels = ax.get_legend_handles_labels()
    handles.remove(handles[7])
    labels.remove(labels[7])
    ax.legend(loc='lower left',handles=handles, labels=labels, ncol=2,fontsize='x-small')
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

stations = list(data.keys())
for station in stations:
    pca_clustering2(data=data,station=station,year1=pd.Timestamp(1989,1,1),year2=pd.Timestamp(2020,1,1),path="E://Summer research/pythonProject/figures/PCA/{}.png".format(station))
#c2 = pca_clustering2(data=data,station='S060211',year1=pd.Timestamp(1989,1,1),year2=pd.Timestamp(2020,1,1))


'''
handles_keep = handles[0:7]
    handles_keep.append(handles[8:15])
    labels_keep = labels[0:7]
    labels_keep.append(labels[8:15])
'''