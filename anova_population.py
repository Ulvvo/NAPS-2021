from DataCleaning import import_pickle,export_pickle
from Classes import Station
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.api as sm
import seaborn as sns
import pylab as py

'''All plots to check distribution'''
def check_normality(arr):
    '''Checks the normality of an array using the Shapiro-Wilk test.
    Input: arr (numpy array) --> numpy array of all the observations you want to test the distribution of.
    Output: stat (float) --> Shapiro-Wilk statistic for normality
            pval (float)  --> P-value for the normality
    '''
    (stat,pval) = scipy.stats.shapiro(arr)
    return (stat,pval)

def create_histogram(voc,cond,arr,start,end):
    '''Plots a histogram of data, given an array.
    Inputs: voc (str) --> name of VOC to create a histogram of
            start (int) --> starting year
            end (int) --> ending year
            cond (str) --> type of condition of the array
            (eg. "LU" for an array of observations for a station categorized as "Large Urbanization")
    Output: A histogram with a kernel density plot showing the distribution of the data points.
    '''
    sns.displot(arr,bins=20,kde=True,color='#607c8e')
    plt.title('{} Conc. Distribution {}-{},{}'.format(voc,start,end,cond))
    plt.xlabel('Concentration (ug/m3)')
    plt.ylabel('Count')
    plt.xticks(np.arange(0,3,0.2))
    plt.xlim([0,3])
    plt.grid(which='minor',axis='x')
    plt.show()

def pplot(arr):
    '''
    Creates a probability plot for an array to test whether it has come from a certain distribution.
    Input: arr (numpy array) --> numpy array of data points
    Output: Probability plot of the data points compared to a given distribution
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    '''In the 'dist' :parameter below, choose the type of distribution. Eg. Exponential, etc.'''
    res = scipy.stats.probplot(arr,dist=scipy.stats.expon,sparams=(2.5,),plot=ax)
    ax.set_title("Q-Q plot for Exponential dist")
    plt.show()

def qqplot(arr):
    '''
    Creates a Q-Q plot for an array to test whether it has come from a certain distribution.
    Input: arr (numpy array) --> numpy array of data points
    Output: Q-Q plot of the data points compared to a given distribution
    '''
    x = (arr-np.nanmean(arr))/np.nanstd(arr) #normalize the data points
    '''Choose the type of distribution in the dist parameter.'''
    sm.qqplot(x,dist=scipy.stats.norm,line='45') #line is '45' because of the normalized data
    py.show()

def normalize(arr):
    '''Transform an array to a normal distribution using a Box Cox transformation.
    Input: arr (numpy array) --> numpy array of data points
    Output: box_cox[0] (numpy array) --> transformed array
    '''
    box_cox = scipy.stats.boxcox(arr)
    return box_cox[0]

'''
How this code works:

1) First get the collapsed data for all the stations & all the years from colldatav3.pickle
2) Group the stations according to the desired condition(s) using group_stations()
3) For the VOCs you want to analyse over a chosen timeframe, use the function verify_vocs() to further filter stations
that contain those VOCs over the chosen years.
4) For each separate list of stations (grouped by filter condition & VOC-verified), combine all their concentration 
arrays into one giant array using the combine_stations() function.
5) Remove outliers in the combined arrays using the remove_outliers_combined() function.
6) Find the Kruskal statistic and P-value using test_ANOVA_kruskal() function.
'''

def group_stations(province,var_condition,categories,metadata,fixed_condition=None,fixed_value=None):
    '''
    Filters NAPS Stations by a variable given condition & optionally fixed condition.

    Takes in an abbreviation of a Canadian province (str), a variable condition for aggregating stations (condition
    values can vary), a list of condition categories, a dataframe containing NAPS station metadata, a fixed condition
    for sorting & a fixed value. Returns a dictionary with stations grouped by each condition of the category.

    Output format: {"<category 1>": [station1, station2,...], "<category 2>": [station8, station9,...],...}

    Eg. Want to group all stations Canada-wide that are "PE" site types by their urbanization level.
    Input: province (str) --> eg. "All" (All provinces) [options: "ON" (Ontario), "QC" (Quebec), "AB" (Alberta), etc.]
           var_condition (str) --> eg. "Urbanization" [options: 'Latitude','Longitude','Neighborhood','Land_use',etc.]
           categories (list) --> eg. ["LU","MU","SU","NU"] for urbanization sizes
           metadata (df) --> eg. import_pickle(<path to metadata df object>)
           fixed_condition (str) --> eg. "Site Type"
           fixed_value (str) --> "PE"

    output: bunch (dict) -->
    eg. {'LU': ['S010102', 'S030113', 'S030117', 'S030118', 'S050103', 'S050104', 'S050121', 'S050122', 'S050124',
    'S050126', 'S050128', 'S050129', 'S050133', 'S050134', 'S050136', 'S050308', 'S060101', 'S060104', 'S060302',
    'S060403', 'S060413', 'S060418', 'S060422', 'S060426', 'S060428', 'S060429', 'S060440', 'S060903', 'S060904',
    'S061502', 'S061602', 'S065101', 'S070119', 'S080110', 'S080111', 'S090130', 'S090227', 'S090228', 'S090230',
    'S100111', 'S100119', 'S100128', 'S100132', 'S100134', 'S101004', 'S101005', 'S101301'], 'MU': ['S061008',
    'S061009', 'S061104', 'S063201', 'S090701', 'S101101'], 'SU': ['S040901', 'S041401', 'S052601', 'S054501',
    'S101701', 'S103202']}
    *if no stations present for a certain category, an empty list is returned
    *for more condition options, see metadata file
    '''
    bunch = {} #create an empty dictionary
    for category in categories:
        if province=='All': #select rows based on province & conditions
            if fixed_condition==None:
                rows = metadata[(metadata[var_condition] == category)]
            else:
                rows = metadata[(metadata[var_condition]==category)&(metadata[fixed_condition]==fixed_value)]
        else:
            if fixed_condition==None:
                rows = metadata[(metadata['Province'] == province) & (metadata[var_condition] == category)]
            else:
                rows = metadata[(metadata['Province']==province)&(metadata[var_condition]==category)&(metadata[fixed_condition]==fixed_value)]
        stations = rows.iloc[:,0].to_list() #convert rows to list
        bunch[category]=stations #add to dictionary
    return bunch

def verify_vocs(bunch,voc,start,end,data):
    '''
    Takes in a dictionary of grouped stations & verifies the stations for which the particular VOC is present
    over the given time period.

    input: bunch (dict) --> the dictionary of grouped stations (eg. {'LU': ['S010102'],'MU':'[S030113', 'S030117'],...}
           voc (str) --> name of VOC (eg. 'Butane')
           start (int) --> start year of measurements (eg. 2001)
           end (int) --> end year of measurements (eg. 2009)
           data (df) --> The dictionary of the complete year-collapsed dataframes for all stations of format:
           {station1:df1, station2:df2, ..., station(i):df(i)},
            where each df is of the format:
            eg. station1: (time goes from years 1989-2019/start-end)
            |   Compounds | VOC1  | VOC2 | ... |
            ----------------------------------------
            |   1/1/1989  | 1.346 | 2.345| ... |
            |    ...      |  ...  |  ... | ... |
            |    ...      |  ...  |  ... | ... |
            |  31/12/2019 | 2.669 | 5.876| ... |

    output: present_stations (dict) --> a dictionary with lists of stations that measure the VOC over the given
    timeframe for each category of the grouped stations condition.

    Eg. For a-Pinene, stations of site type "PE" for different Urbanization categories from 2003-2013
    {'LU': ['S010102', 'S030118', 'S050103', 'S050104', 'S050121', 'S050129', 'S050133', 'S050134', 'S060101',
    'S060104', 'S060302', 'S060413', 'S060418', 'S060428', 'S060429', 'S060903', 'S061502', 'S061602', 'S065101',
    'S070119', 'S080110', 'S090130', 'S090227', 'S090228', 'S100111', 'S100119', 'S100128', 'S100134', 'S101004',
    'S101005'], 'MU': ['S063201', 'S101101'], 'SU': ['S054501', 'S101701', 'S103202']}

    '''
    present_stations = {}
    for category in bunch:
        present_stations[category]=[]
        for station in bunch[category]:
            stationobj = Station(station,data) #create a 'Station' object
            try: #if the VOC is present for the station over all the years, then it can be accessed
                arr = stationobj.get_voc(voc,start,end) #get the 'Compounds' and VOC conc df
                if (len(arr)!=0) &  (arr.iloc[:,1].isnull().all()==False): #ensure the array is not empty & not all NaN
                    present_stations[category].append(station)
            except KeyError: #if the VOC is not present at all, ignore
                pass
    return present_stations

'''Works for any categories that have stations present. Still works when some categories have empty stations.'''
def combine_stations(stationlist,voc,start,end,data):
    '''Takes a list of stations and combines their all their concentration arrays into one big array.

    inputs: stationlist (list) --> list of stations to combine (eg. ['S100100','S030118',...])
           voc (str) --> name of VOC (eg. 'Butane')
           start (int) --> starting year (eg. 1999)
           end (int) --> ending year (eg. 2005)
           data (dict of dfs) --> The dictionary of the complete year-collapsed dataframes for all stations of format:
           {station1:df1, station2:df2, ..., station(i):df(i)},
            where each df is of the format:
            eg. station1: (time goes from years 1989-2019/start-end)
            |   Compounds | VOC1  | VOC2 | ... |
            ----------------------------------------
            |   1/1/1989  | 1.346 | 2.345| ... |
            |    ...      |  ...  |  ... | ... |
            |    ...      |  ...  |  ... | ... |
            |  31/12/2019 | 2.669 | 5.876| ... |

    output: combined (df) --> a df of all combined concentration observations of a given VOC over a given period
    of time for a list of stations.

    eg. |   Compounds | VOC   |
        -----------------------
        |   1/1/1999  | 1.346 |
        |    ...      |  ...  |
        |    ...      |  ...  |
        |  31/12/2005 | 2.669 |

    '''
    dflist=[] #create an empty list
    if len(stationlist)!=0:
        for station in stationlist:
            stationobj = Station(station,data) #create a Station object
            df = stationobj.get_voc(voc,start,end) #get the time & voc conc df
            dflist.append(df) #add the df to the dflist to join
        combined = pd.concat(dflist,axis=0,join='outer',sort=True) #concatenate all the dfs in the dflist
        combined.sort_values(by=['Compounds'],ascending=True,inplace=True) #get in chronological order
        return combined
    else:
        return None

'''Does not work on None-type from previous function'''
def remove_outliers_combined(combined,voc):
    '''
    Takes a combined dataframe of stations and returns a concentration array with the most extreme outliers removed.
    Empirical rule used to remove data points 3 standard deviations away from the mean.

    input: combined (df) --> a dataframe of time & VOC conc. combined for different stations.
    Eg. Combined df of stations for a VOC that started measurements in 1999 and ended 2005.
        |   Compounds | VOC   |
        -----------------------
        |   1/1/1999  | 1.346 |
        |    ...      |  ...  |
        |    ...      |  ...  |
        |  31/12/2005 | 2.669 |

    output: new_arr (numpy array) --> an array of measurements with outliers removed
    eg. [1.34 1.22 2.75 2.08 ...]
    '''
    arr = combined[voc]
    xbar = np.nanmean(arr)
    std = np.nanstd(arr)
    '''Using the Empirical Rule --> 99.7% of the data will fall within 3 standard deviations of the mean'''
    upperlim = xbar + 3*std
    lowerlim = xbar - 3*std
    new_arr = arr[(arr>=lowerlim)&(arr<=upperlim)]
    return new_arr

def test_ANOVA_kruskal(present_stations,voc,start,end,data):
    '''Computes a Kruskal-Wallis statistic for different arrays'''
    arrays = []
    for condition in present_stations:
        stationslist = present_stations[condition] #some station lists might be empty
        combined = combine_stations(stationslist,voc,start,end,data) #combines all non-empty station lists
        if isinstance(combined,pd.DataFrame)==True:#only removes outliers for combined dataframes that are non-None objects
            new_arr = remove_outliers_combined(combined,voc)
            arrays.append(new_arr)
    if len(arrays)!=0:
        kruskal = scipy.stats.kruskal(*arrays,nan_policy='omit')
        return (kruskal,arrays)
    else:
        return None

def test_ANOVA_F(present_stations,voc,start,end,data):
    '''Computes a Kruskal-Wallis statistic for different arrays'''
    arrays = []
    for condition in present_stations:
        stationslist = present_stations[condition]
        combined = combine_stations(stationslist,voc,start,end,data)
        new_arr = remove_outliers_combined(combined,voc)
        new_arr = normalize(new_arr)
        arrays.append(new_arr)
    F_test = scipy.stats.f_oneway(arrays[0],arrays[1],arrays[2])
    return F_test, arrays

def calc_stats(voc,province,var_condition,categories,metadata,fixed_condition=None,fixed_value=None):
    '''Calculates the summary statistics, and Kruskal-Wallis test for each VOC over all the decades.'''
    start = [1989,1999,2009] #get the start years of each decade
    grouped_stations = group_stations(province,var_condition,categories,metadata,fixed_condition,fixed_value) #group stations by a condition
    df_dict = {'Year':[],'Type':[],'Stations':[],'N':[],'Mean':[],'Median':[],'Std':[],'Critical Value':[],'kruskal_stat':[],'kruskal_val':[]}
    for year in start: #do this process for every decade
        if year==1989 or year==1999:
            end = year+10
        else:
            end=year+11
        present_stations = verify_vocs(grouped_stations, voc, year, end, data)
        kruskal_results = test_ANOVA_kruskal(present_stations,voc,year,end,data)
        if kruskal_results!= None:
            kruskal_stat = kruskal_results[0].statistic
            kruskal_p = kruskal_results[0].pvalue
            arrays = kruskal_results[1]
        else:
            kruskal_stat=''
            kruskal_p=''
            arrays=[]
        i=0
        for site in categories:
            if len(arrays)!=0:
                arr = arrays[i]
                N = len(arr)
                mean = np.nanmean(arr)
                median = np.nanmedian(arr)
                sd = np.nanstd(arr)
                critical_val = scipy.stats.chi2.ppf(q=0.95,df=len(categories)-1)
                df_dict['Year'].append(year)
                df_dict['Type'].append(site)
                df_dict['Stations'].append(present_stations[site])
                df_dict['N'].append(N)
                df_dict['Mean'].append(mean)
                df_dict['Median'].append(median)
                df_dict['Std'].append(sd)
                df_dict['Critical Value'].append(critical_val)
                df_dict['kruskal_stat'].append(kruskal_stat)
                df_dict['kruskal_val'].append(kruskal_p)
            i+=1
    df = pd.DataFrame(df_dict)
    return df

'''Running zone'''
# md = import_pickle("E:\Summer research\pythonProject\data\metadatav2.pickle")[0]
# bs = group_stations("All","Urbanization",["LU","MU","SU"],md)
# data = import_pickle("E://Summer research/pythonProject/data/colldatav3.pickle")[0]
# df = calc_stats('a-Pinene','All',"Urbanization",["LU","MU","SU"],md)
