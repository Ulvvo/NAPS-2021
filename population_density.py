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

linear_reg_df = pd.read_excel("E:\Summer research\pythonProject\data\Linear Regression.xlsx")
voc = 'Dodecane'
def pop_density(voc):
    years = [(1989, 1994), (1994, 1999), (1999, 2004),(2004,2009),(2009,2014),(2014,2020)]
    x = np.arange(0,4000,100)
    for start,end in years:
        m = linear_reg_df['b1: Pop. density'][(linear_reg_df['VOC'] == voc) & (linear_reg_df['Year'] == start)].iloc[0]
        c = linear_reg_df['b0: constant'][(linear_reg_df['VOC'] == voc) & (linear_reg_df['Year'] == start)].iloc[0]
        l1 = linear_reg_df['lambda (x)'][(linear_reg_df['VOC'] == voc) & (linear_reg_df['Year'] == start)].iloc[0]
        l2 = linear_reg_df['lambda (y)'][(linear_reg_df['VOC'] == voc) & (linear_reg_df['Year'] == start)].iloc[0]
        l2_l1 = linear_reg_df['ly/lx'][(linear_reg_df['VOC'] == voc) & (linear_reg_df['Year'] == start)].iloc[0]
        y = (m * l2_l1 * (x ** l1) + c * l2 - m * l2_l1 + 1) ** (1 / l2)
        sns.lineplot(x=x,y=y,label='{}'.format(start),palette='Set2')
    plt.xlabel('Population density (people/sq km)',fontdict={'family':'serif','size':10})
    plt.ylabel('{} concentration (ug/m3)'.format(voc),fontdict={'family':'serif','size':10})
    plt.title('{} concentration vs. Population density'.format(voc),fontdict={'family':'serif','size':12})
    plt.show()

# selected_df = linear_reg_df[linear_reg_df['VOC']==voc]
# df = pd.DataFrame(columns=['Population density (people/sq km)','1989-1994','1994-1999','1999-2004','2004-2009',
#                            '2009-2014','2014-2020'])
# x = np.linspace(0,4000,100)
# df['Population density (people/sq km)']=x
# for start,end in years:
#     try:
#         row = selected_df[selected_df['Year']==start].iloc[0]
#         m = row['b1: Pop. density']
#         lambda1 = row['lambda (x)']
#         lambda2 = row['lambda (y)']
#         c = row['b0: constant']
#         y =
#         df['{}-{}'.format(start,end)]=y
#     except KeyError:
#         pass
