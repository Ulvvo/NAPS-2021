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

'''This file involves some initial analysis of the data including finding time trends, drawing distributions with 
box plots & KDEs.'''

def clean_array(df,voc):
    '''
    Takes an array and returns it with the most extreme outliers removed.
    Empirical rule used to remove data points 3 standard deviations away from the mean.

    input: arr (numpy array) --> an array of concentration measurements

    output: new_arr (numpy array) --> an array of measurements with outliers removed
    eg. [1.34 1.22 2.75 2.08 ...]
    '''
    xbar = np.nanmean(df[voc])
    std = np.nanstd(df[voc])
    '''Using the Empirical Rule --> 99.7% of the data will fall within 3 standard deviations of the mean'''
    upperlim = xbar + 3*std
    lowerlim = xbar - 3*std
    new_df = df[(df[voc]>=lowerlim)&(df[voc]<=upperlim)]
    return new_df[(new_df!=0)&(new_df!=np.nan)]

def time_trend_singular(alldata,station,voc,ylim,window=12,path=None):
    '''
    Creates a time series plot for a single station & VOC compound.

    input: df (Pandas DataFrame) of a given station of the format:
    eg. station1: (time goes from the very beginning of measurements to the very end)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

        voc (str) --> name of the VOC (eg. 'Ethane')

    output: Time series plot (using seaborn) for that station and that VOC over all the years
    '''
    df = alldata[station]
    fig = plt.figure(figsize=(20,10)) #create a figure object
    sns.set_color_codes('pastel') #set palette colour codes
    ax = sns.lineplot(x="Compounds",y=voc,data=df[['Compounds',voc]],color='b') #create a lineplot using Seaborn
    #seasonality: window=12
    df['smoothed']=df[voc].rolling(window=window).median()
    sns.lineplot(ax=ax,x="Compounds",y='smoothed',data=df,color='darkorchid',linewidth=4)
    sns.color_palette('pastel') #set the colour palette
    ax.set_xlabel('Year') #date on the x axis
    ax.set_ylabel('Concentration of {} (ug/m3)'.format(voc)) #concentration on the y-axis
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(1, month=1, day=1))  # have major ticks every year
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1))  # have minor ticks every month
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))  # this shows the major ticks as just years instead of timestamps
    ax.set_xticks(np.arange(df['Compounds'].iloc[0].date(), df['Compounds'].iloc[-1].date()), minor=True)
    handles = [mpl.lines.Line2D([], [], color='b', label='Raw'),
               mpl.lines.Line2D([], [], color='darkorchid', label='Smoothed')]
    ax.legend(loc='upper right', title='Time series', handles=handles, fancybox=False)
    #ax.set_xticks(np.arange(df['Compounds'].iloc[0].date(),df['Compounds'].iloc[-1].date()),minor=True) #set ticks
    ax.set_ylim(0,ylim)
    plt.minorticks_on()
    fig.suptitle("Concentration of {} over time, station {}".format(voc,station))
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

def time_trends(stationlist,voc,province,alldata):
    '''Creates a time series plot for all the stations of a given province for a given VOC. Each station shows up as a
    line in the graph and is shown in the legend.

    inputs: stationlist (list) --> list of stations for that province (eg. ['S100110','S100111',...])
            voc (str) --> name of VOC (eg. 'Ethane')
            province (str) --> name of province (eg. 'Ontario')
            alldata (dict of dfs) --> a dictionary of Dataframes containing VOC data for all stations over the entire
            time period of measurements. Of the form: {station1:df1, station2:df2, ..., station(i):df(i)},
            where each df is of the format:
            eg. station1: (time goes from years 1989-2019) or whichever is the start and end time for observations
            |   Compounds | VOC1  | VOC2 | ... |
            ----------------------------------------
            |   1/1/1989  | 1.346 | 2.345| ... |
            |    ...      |  ...  |  ... | ... |
            |    ...      |  ...  |  ... | ... |
            |  31/12/2019 | 2.669 | 5.876| ... |

    output: a Seaborn Time Series plot of different stations each represented as lines with a legend for a given VOC
    and province.
    '''
    fig, ax = plt.subplots(figsize=(20,10)) #create a figure and axes object
    y_lims = [] #create an empty list of y limits of the graph
    for station in stationlist:
        df = alldata[station] #get the df for each station in the stationlist
        try: #try get a cleaned df (i.e. without outliers) for the given VOC
            df = clean_array(df, voc)
            #if the VOC is present, a df is returned and then create a time series for it
            sns.lineplot(ax=ax,x='Compounds',y=voc,data=df,legend='brief',label=station,palette='Set2')
            upper_ylim = round(df[voc].max()) #get the upper y-limit of the concentration array & round it
            if upper_ylim == 0: #if the rounded y-limit is 0, then take the original maximum and get a limit 10% higher
                upper_ylim=1.1*(df[voc].max())
            y_lims.append(upper_ylim) #add the y-limit to the y_lims list for reference
        except:
            pass #if the VOC is not present in the DF then do nothing - no need to add a time series
    #set labels & format
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration (ug/m3)')
    ax.set_ylim(ymin=0, ymax=max(y_lims)) #maximum y value on graph is the max value of the y_lims list
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(1, month=1, day=1)) #have major ticks every year
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1)) #have minor ticks every month
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y')) #this shows the major ticks as just years instead of timestamps
    ax.set_xticks(np.arange(df['Compounds'].iloc[0].date(), df['Compounds'].iloc[-1].date()), minor=True)
    ax.legend()
    plt.minorticks_on()
    fig.suptitle("Concentration of {} over time in {}".format(voc,province))
    plt.show()

def draw_graphs(stationlist,province,alldata,voc_list):
    '''
    Produces time trend graphs for multiple VOCs in a given list.

    inputs: stationlist (list) --> a list of stations present within a given province
            province (str) --> name of province (eg. 'Ontario')
            alldata (dict of dfs) --> a dictionary of Dataframes containing VOC data for all stations over the entire
            time period of measurements. Of the form: {station1:df1, station2:df2, ..., station(i):df(i)},
            where each df is of the format:
            eg. station1: (time goes from years 1989-2019) or whichever is the start and end time for observations
            |   Compounds | VOC1  | VOC2 | ... |
            ----------------------------------------
            |   1/1/1989  | 1.346 | 2.345| ... |
            |    ...      |  ...  |  ... | ... |
            |    ...      |  ...  |  ... | ... |
            |  31/12/2019 | 2.669 | 5.876| ... |
            voclist (list) --> a list of VOCs to draw graphs for (eg. ['Ethane','Ethylene',...])
    output: a Seaborn Time Series plot of different stations each represented as lines with a legend for a given VOC
    and province. Multiple time series returned for different VOCs.
    '''
    for voc in voc_list:
        time_trends(stationlist,voc,province,alldata)

def find_ylim_boxplots(combined_provinces):
    '''
    Takes a DataFrame of combined stations and returns a value for the y-limit of the concentration axis.
    ***BASED ON THE EMPIRICAL RULE OF GAUSSIAN DISTRIBUTION***

    input: combined_provinces (df) --> a DataFrame showing concentrations of a given VOC for different stations as columns
    over the entire period of measurement time.

    eg. VOC1: (time goes from whichever is the start and end time for observations)
    |   Compounds | Station1  | Station2 | ... |
    --------------------------------------------
    |   1/1/1989  |   1.346   |   2.345  | ... |
    |    ...      |    ...    |    ...   | ... |
    |    ...      |    ...    |    ...   | ... |
    |  31/12/2019 |   2.669   |   5.876  | ... |

    output: ylim (float) --> value of the y-limit (eg. 10.5)
    '''
    upper_lims = []
    for station in combined_provinces.columns.to_list()[1:]:
        xbar = np.nanmean(combined_provinces[station])
        std = np.nanstd(combined_provinces[station])
        '''Using the Empirical Rule --> 99.7% of the data will fall within 3 standard deviations of the mean'''
        upperlim = xbar + 2.8*std
        upper_lims.append(upperlim)
    maximum = max(upper_lims)
    if maximum<0.5:
        ylim = 1.1*maximum
    else:
        ylim = round(maximum)
    return ylim

def find_ylim_boxplots2(combined_provinces):
    '''
    Takes a DataFrame of combined stations and returns a value for the y-limit of the concentration axis.
    ***BASED ON THE INTER-QUARTILE RANGE OF THE BOXPLOTS***

    input: combined_provinces (df) --> a DataFrame showing concentrations of a given VOC for different stations as columns
    over the entire period of measurement time.

    eg. VOC1: (time goes from whichever is the start and end time for observations)
    |   Compounds | Station1  | Station2 | ... |
    --------------------------------------------
    |   1/1/1989  |   1.346   |   2.345  | ... |
    |    ...      |    ...    |    ...   | ... |
    |    ...      |    ...    |    ...   | ... |
    |  31/12/2019 |   2.669   |   5.876  | ... |

    output: ylim (float) --> value of the y-limit (eg. 10.5)
    '''
    upper_lims = []
    for station in combined_provinces.columns.to_list()[1:]:
        lower_quartile = np.nanpercentile(combined_provinces[station],25)
        upper_quartile = np.nanpercentile(combined_provinces[station], 75)
        IQR = upper_quartile - lower_quartile
        max_line = 1.5*IQR+upper_quartile
        upper_lims.append(max_line)
    maximum = np.nanmax(upper_lims)
    ylim = 1.05*maximum
    return ylim

'''For Series'''
def find_ylim_boxplots3(series):
    '''
    Takes a DataFrame of combined stations and returns a value for the y-limit of the concentration axis.
    ***BASED ON THE INTER-QUARTILE RANGE OF THE BOXPLOTS***

    input: combined_provinces (df) --> a DataFrame showing concentrations of a given VOC for different stations as columns
    over the entire period of measurement time.

    eg. VOC1: (time goes from whichever is the start and end time for observations)
    |   Compounds | Station1  | Station2 | ... |
    --------------------------------------------
    |   1/1/1989  |   1.346   |   2.345  | ... |
    |    ...      |    ...    |    ...   | ... |
    |    ...      |    ...    |    ...   | ... |
    |  31/12/2019 |   2.669   |   5.876  | ... |

    output: ylim (float) --> value of the y-limit (eg. 10.5)
    '''
    upper_lims = []
    lower_quartile = np.nanpercentile(series,25)
    upper_quartile = np.nanpercentile(series, 75)
    IQR = upper_quartile - lower_quartile
    max_line = 1.5*IQR+upper_quartile
    upper_lims.append(max_line)
    maximum = max(upper_lims)
    ylim = 1.3*maximum
    return ylim

def clean_df(combined_provinces):
    '''
    Takes a DataFrame of combined stations and returns it with the most extreme outliers removed.
    Empirical rule used to remove data points 3 standard deviations away from the mean.

    input: combibed_provinces (df) --> a DataFrame showing concentrations of a given VOC for different stations as
    columns over the entire period of measurement time.
    eg. VOC1: (time goes from whichever is the start and end time for observations)
    |   Compounds | Station1  | Station2 | ... |
    --------------------------------------------
    |   1/1/1989  |   1.346   |   2.345  | ... |
    |    ...      |    ...    |    ...   | ... |
    |    ...      |    ...    |    ...   | ... |
    |  31/12/2019 |   2.669   |   5.876  | ... |

    output: new_df (Pandas DataFrame) --> combined_provinces with outliers removed
    '''
    new_df = pd.DataFrame(columns=['Compounds'])
    for station in combined_provinces.columns.to_list()[1:]:
        xbar = np.nanmean(combined_provinces[station])
        std = np.nanstd(combined_provinces[station])
        '''Using the Empirical Rule --> 99.7% of the data will fall within 3 standard deviations of the mean'''
        upperlim = xbar + 3*std
        selected_df = combined_provinces[['Compounds',station]][combined_provinces[station]<=upperlim]
        new_df=new_df.merge(selected_df,how='outer',on=['Compounds'])
    new_df.sort_values(by='Compounds',ascending=True,inplace=True)
    return new_df

def create_boxplots(province_data,voc):
    '''
    Creates a figure of multiple box plots for different stations of different provinces for a given VOC.

    *** NEED to use 10 yrs+ of data (see: select_10() function in DataCleaning.py)
    input: province_data (dict) --> Dictionary containing Pandas dataframes & station names for each province.
    Structure: NAPS province code as keys, and lists of tuples as values. In each tuple, the first element is the df,
    and the second element is the station name.

    eg. province_data = {1: [(df10102,'S010102')], 3: [(df30112,'S030113'),(df30118,'S030118'),...],
    4: [(df40203,'S040203'),...],5: [(df50115,'S050115'),...], 6: [...], 7: [...], 8: [...], 9: [...], 10: [...]}

    voc (str) --> name of VOC (eg. 'Ethane')

    Output: multiple boxplots of different stations and multiple provinces.
    Each row --> represents a new province.
    Each column --> represents a new decade.
    Different stations are plotted on the same graph.

    '''
    n_rows = 3 #define number of rows (based on number of provinces)
    n_cols = 3 #define number of cols (based on the decades)
    #create a Figure & Axes object with given number of rows & columns & chosen figure size
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(25,20))
    i=0 #the initial row index is 0
    #create a dictionary of province NAPS code and names
    provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
    y_lims = [] #create an empty dictionary of y limits
    for province in province_data:
        province_combined = combine_df_province(province,voc,province_data)
        decades = [(1989,1999),(1999,2009),(2009,2020)]
        j = 0
        for start,end in decades:
            start_yr = pd.to_datetime('{}-01-01'.format(start),infer_datetime_format=True)
            end_yr = pd.to_datetime('{}-01-01'.format(end),infer_datetime_format=True)
            df_time = province_combined[(province_combined['Compounds']>=start_yr) & (province_combined['Compounds']<end_yr)]
            df_time.dropna(axis=1, how='all', inplace=True)
            df_time = clean_df(df_time)
            try:
                sns.boxplot(ax=axes[i, j], data=df_time,palette='Set2')
                font = {'family': 'serif',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 10,
                        }
                axes[i,j].set_title('{}, years {} - {}'.format(provinces[province],start_yr.year,end_yr.year),fontdict=font)
                axes[i,j].set_ylim(ymin=0,ymax=find_ylim_boxplots(df_time))
                y_lims.append(find_ylim_boxplots(df_time))
                axes[i,j].set_xlabel('Stations',fontdict=font)
                axes[i,j].set_ylabel('Concentration (ug/m3)',fontdict=font)
                axes[i,j].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(10))
            except ValueError:
                pass
            j += 1
        i+=1
    font_title = {'family': 'serif',
            'weight': 'bold',
            'size': 30,
            }
    fig.suptitle("Box plots of {} concentration across decades & provinces".format(voc),y=0.98,fontproperties=font_title)
    plt.show()
    return y_lims

'''
This is for Quebec, Ontario and B.C.

Options:
1) To see outliers & only omit data 3 stds away, use find_ylim_boxplots().
2) To see a closer view of stations (y limit just above the upper bar of the box plot), use find_ylim_boxplots2().
3) Can modify whether the figure is saved or showed.

'''
def create_boxplots2(province_data,voc,path):
    '''
    Creates a figure of multiple box plots for different stations of different provinces for a given VOC.

    *** NEED to use 10 yrs+ of data (see: select_10() function in DataCleaning.py)
    input: province_data (dict) --> Dictionary containing Pandas dataframes & station names for each province.
    Structure: NAPS province code as keys, and lists of tuples as values. In each tuple, the first element is the df,
    and the second element is the station name.

    **Only contains one province
    eg. province_data = {5: [(df50101,'S050101'),...]}

    voc (str) --> name of VOC (eg. 'Ethane')

    Output: multiple boxplots of different stations for different decades. 3 rows, 1 column.
    Each row --> represents a new decade.
    Different stations are plotted on the same graph.

    '''
    n_rows = 3 #define number of rows (based on number of provinces)
    n_cols = 1 #define number of cols (based on the decades)
    #create a Figure & Axes object with given number of rows & columns & chosen figure size
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(30,20))
    provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
    y_lims = []
    for province in province_data: #get the list of tuples
        #combine all the stations in the list into one singular dataframe
        province_combined = combine_df_province(province,voc,province_data)
        decades = [(1989,1999),(1999,2009),(2009,2020)]
        i = 0
        for start,end in decades:
            start_yr = pd.to_datetime('{}-01-01'.format(start),infer_datetime_format=True)
            end_yr = pd.to_datetime('{}-01-01'.format(end),infer_datetime_format=True)
            df_time = province_combined[(province_combined['Compounds']>=start_yr) & (province_combined['Compounds']<end_yr)]
            df_time.dropna(axis=1, how='all', inplace=True)

            try:
                sns.boxplot(ax=axes[i], data=df_time,palette='Set2')
                font = {'family': 'serif',
                        'color': 'black',
                        'weight': 'normal',
                        'size': 14,
                        }
                axes[i].set_title('{}, years {} - {}'.format(provinces[province],start_yr.year,end_yr.year),fontdict=font)
                axes[i].set_ylim(ymin=0,ymax=find_ylim_boxplots2(df_time))
                y_lims.append(find_ylim_boxplots2(df_time))
                axes[i].set_xlabel('Stations',fontdict=font)
                axes[i].set_ylabel('Concentration (ug/m3)',fontdict=font)
                axes[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(10))
            except ValueError:
                pass
            i += 1
    font_title = {'family': 'serif',
            'weight': 'bold',
            'size': 14,
            }
    fig.suptitle("Box plots of {} concentration across decades & provinces".format(voc),y=0.983,fontproperties=font_title)
    plt.show()
    # plt.savefig(path)
    return y_lims

'''This is a different view of boxplots'''
def create_boxplots3(province_data,voc,path=None):
    '''
    Creates a figure of multiple box plots for different stations of different provinces for a given VOC.

    *** NEED to use 10 yrs+ of data (see: select_10() function in DataCleaning.py)
    input: province_data (dict) --> Dictionary containing Pandas dataframes & station names for each province.
    Structure: NAPS province code as keys, and lists of tuples as values. In each tuple, the first element is the df,
    and the second element is the station name.

    **Only contains one province
    eg. province_data = {5: [(df50101,'S050101'),...]}

    voc (str) --> name of VOC (eg. 'Ethane')

    Output: multiple boxplots of different stations for different decades. 3 rows, 1 column.
    Each row --> represents a new decade.
    Different stations are plotted on the same graph.

    '''
    province_int = [*province_data][0]
    n_rows = len(province_data[province_int]) #define number of rows (based on number of stations within province)
    n_cols = 3 #define number of cols (based on the decades)
    #create a Figure & Axes object with given number of rows & columns & chosen figure size
    fig, axes = plt.subplots(nrows=n_rows,ncols=n_cols,figsize=(20,30))
    provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
    y_lims = []
    for province in province_data: #get the list of tuples
        #combine all the stations in the list into one singular dataframe
        province_combined = combine_df_province(province,voc,province_data)
        try: #to test whether province_combined is a DataFrame, get the first item
            first_item = province_combined.iloc[0,0]
        except AttributeError: #if the province_combined is None, then attribute error is thrown
            raise Exception('None of the stations in this province have data for {}'.format(voc))
        decades = [(1989,1999),(1999,2009),(2009,2020)] #get the start and end of each decade
        j = 0 #start off with the first column
        df_period_stations={} #create an empty dictionary for the periods
        for start,end in decades:
            start_yr = pd.to_datetime('{}-01-01'.format(start),infer_datetime_format=True)
            end_yr = pd.to_datetime('{}-01-01'.format(end),infer_datetime_format=True)
            #for each decade, get the dataframe for that range only
            df_time = province_combined[(province_combined['Compounds']>=start_yr) & (province_combined['Compounds']<end_yr)]
            #remove all zero values & replace with nan, drop nan
            df_time=df_time.replace(to_replace=0,value=np.nan)
            df_time.dropna(axis=1, how='all', inplace=True)
            #add the list of stations present for that period
            df_period_stations[start]=df_time.columns.to_list()[1:]
            i=0 #start off with the first row
            for station in df_time.columns.to_list()[1:]:
                if j!=0: #if we are on the second column onwards, need to ensure we are on the same station for that row
                    try: #try get the row value of the same station but one decade prior (i.e. the previous col)
                        i = df_period_stations[start-10].index(station)
                    except ValueError: #if the first column had no graph for that station, then move on
                        pass
                try:
                    sns.boxplot(ax=axes[i,j],x=station,orient='h',data=df_time, palette='Set2')
                    font = {'family': 'serif',
                            'color': 'black',
                            'weight': 'normal',
                            'size': 14,
                            }
                    axes[i,j].set_title('Station {}, years {} - {}'.format(station, start_yr.year, end_yr.year),
                                      fontdict=font)
                    axes[i,j].set_xlim(xmin=0, xmax=find_ylim_boxplots3(df_time[station]))
                    y_lims.append(find_ylim_boxplots3(df_time[station]))
                    axes[i,j].set_ylabel('Stations', fontdict=font)
                    axes[i,j].set_xlabel('Concentration (ug/m3)', fontdict=font)
                    axes[i,j].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(10))
                    # axes[i,j].xaxis.set_minor_formatter(mpl.ticker.FixedFormatter())
                except TypeError:
                    raise Exception('x variable is non-numeric for station {}, voc {}, decade {}'.format(station,voc,start))
                except ValueError:
                    pass
                i += 1
            j+=1
    font_title = {'family': 'serif',
            'weight': 'bold',
            'size': 14,
            }
    fig.suptitle("Box plots of {} concentration across decades for {}".format(voc,provinces[province_int]),y=0.992,fontproperties=font_title)
    fig.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
    return y_lims

def group_by_decades(province_combined,station,decades):
    '''Returns a DataFrame for a station with each column as values for a given decade'''
    df_station = province_combined[['Compounds',station]]
    start = decades[0][0]
    end = decades[0][1]
    start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
    end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
    df_time = province_combined[['Compounds', station]][
        (province_combined['Compounds'] >= start_yr) & (province_combined['Compounds'] < end_yr)]

'''This is another view of the boxplots for Ontario, B.C. and Quebec.'''
def create_boxplots4(province_data,voc,path=None):
    '''
    Creates a figure of multiple box plots for different stations of different provinces for a given VOC.

    *** NEED to use 10 yrs+ of data (see: select_10() function in DataCleaning.py)
    input: province_data (dict) --> Dictionary containing Pandas dataframes & station names for each province.
    Structure: NAPS province code as keys, and lists of tuples as values. In each tuple, the first element is the df,
    and the second element is the station name.

    **Only contains one province
    eg. province_data = {5: [(df50101,'S050101'),...]}

    voc (str) --> name of VOC (eg. 'Ethane')

    Output: multiple boxplots of different stations for different decades. 3 rows, 1 column.
    Each row --> represents a new decade.
    Different stations are plotted on the same graph.

    '''
    province_int = [*province_data][0]
    n_stations = len(province_data[province_int]) #find the number of stations present for the province
    n_cols = n_stations #define number of cols (based on fixed arrangement)
    n_rows = math.ceil(n_stations/n_cols)#define number of rows based on fixed number of cols
    #create a Figure & Axes object with given number of rows & columns & chosen figure size
    fig, axes = plt.subplots(nrows=1,ncols=n_cols,figsize=(60,10))
    provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
    i=0
    for (df,station) in province_data[province_int]: #unpack the df, station name tuples
        '''Choose based on year intervals'''
        # decades = [(1989, 1999), (1999, 2009), (2009, 2020)]  # get the start and end of each decade
        decades = [(1989, 1994), (1994, 1999), (1999, 2004),(2004,2009),(2009,2014),(2014,2020)]
        start = decades[0][0]
        end = decades[0][1]
        start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
        end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
        try: #do the following if the voc is present in the data for that station
            df_times = df[['Compounds',voc]][(df['Compounds'] >= start_yr) & (df['Compounds'] < end_yr)]
            '''Choose based on year intervals'''
            # df_times=df_times.rename(columns={voc:'1989\n1999'})
            df_times = df_times.rename(columns={voc: '1989\n1994'})
            df_times=df_times.replace(to_replace=0,value=np.nan) #Do not do anything to the zero values
            for start,end in decades[1:]:
                start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
                end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
                df_decade = df[['Compounds', voc]][(df['Compounds'] >= start_yr) & (df['Compounds'] < end_yr)]
                df_decade = df_decade.rename(columns={voc: '{}\n{}'.format(start,end)})
                df_decade = df_decade.replace(to_replace=0, value=np.nan)
                df_times=df_times.merge(df_decade,how='outer',on=['Compounds'])
            '''Choose based on year intervals'''
            #df_times = df_times[['Compounds', '1989\n1999', '1999\n2009', '2009\n2020']]
            df_times = df_times[['Compounds', '1989\n1994', '1994\n1999', '1999\n2004','2004\n2009', '2009\n2014','2014\n2020']]
            sns.boxplot(ax=axes[i],data=df_times,palette='Set2')
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 14,
                    }
            urbanization=(metadata['Urbanization'][metadata['NAPS ID'] == station]).iloc[0]
            axes[i].set_title('Station {}\n{}'.format(station,urbanization),fontdict=font)
            ymax = find_ylim_boxplots2(df_times)
            axes[i].set_ylim(ymin=0, ymax= ymax)
            axes[i].set_ylabel('Concentration (ug/m3)', fontdict=font)
            axes[i].set_xlabel('Years', fontdict=font)
            axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
            axes[i].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            i+=1
        except KeyError: #voc not found at all in the station
            pass
    font_title = {'family': 'serif',
                'weight': 'bold',
                'size': 14,
                }
    fig.suptitle("Box plots of {} concentration across decades for {}".format(voc,provinces[province_int]),y=0.992,fontproperties=font_title)
    fig.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)


'''Running zone'''
alldata = import_pickle("E://Summer research/pythonProject/data/colldatav4.pickle")[0]
prov_data = import_pickle("E://Summer research/pythonProject/data/10_years_provinces.pickle")[0]
#allvocs = import_pickle('E://Summer research/pythonProject/data/allvocs.pickle')[0]
metadata = import_pickle('E://Summer research/pythonProject/data/metadata.pickle')[0]

prov_sel4 = [6]
selected_provinces = {prov:prov_data[prov] for prov in prov_sel4}

VCPs = ['d-Limonene','Camphene','Acetone','Propane','Butane','Isopentane','Cyclopentane','Pentane','Cyclohexane',
        'Heptane','Methylcyclohexane','Toluene','Octane','Ethylbenzene','Styrene','Decane','Undecane','Dodecane',
        'Bromomethane','Dichloromethane','Tetrachloroethylene','1,4-Dichlorobenzene','1,3-Dichlorobenzene',
        '1,2-Dichlorobenzene','MIBK']

# create_boxplots4(selected_provinces,VCPs[-2])
# stationlist = [station for (df,station) in selected_provinces[6]]
# time_trends(stationlist,'Dichloromethane','Ontario',alldata)
# for voc in VCPs:
#     create_boxplots4(selected_provinces,voc,"E://Summer research/pythonProject/figures/boxplots/Alberta/View2/{}.png".format(voc))
