def combine_df_province(province,voc,province_data):
    '''
    Combines data for different stations in a given province into a single DF for a given VOC with the columns being
    measurements for different stations.

    input: province (int) --> NAPS Station code for that province (eg. 6 --> Ontario). *See function group_by_province()
    for a full legend.

    voc (str) --> name of VOC (eg. 'Ethane')
    province_data (dict of lists containing tuples) --> each province key has a list containing tuples with the
    first element being the dataframe for that station & the second element being the name of station.

    eg. province_data = {1: [(df10102,'S010102')], 3: [(df30112,'S030113'),(df30118,'S030118'),...],
    4: [(df40203,'S040203'),...],5: [(df50115,'S050115'),...], 6: [...], 7: [...], 8: [...], 9: [...], 10: [...]}

    output: combined_df (df) --> a DataFrame showing concentrations of a given VOC for different stations as columns
    over the entire period of measurement time.

    eg. VOC1: (time goes from whichever is the start and end time for observations)
    |   Compounds | Station1  | Station2 | ... |
    --------------------------------------------
    |   1/1/1989  |   1.346   |   2.345  | ... |
    |    ...      |    ...    |    ...   | ... |
    |    ...      |    ...    |    ...   | ... |
    |  31/12/2019 |   2.669   |   5.876  | ... |
    '''
    df_pair_list = province_data[province]
    #get the first df for the reference df to merge with
    first_df,first_stationname = df_pair_list[0]
    first_present = False
    try: #try see if the first df has the voc we need
        first_df = first_df[['Compounds',voc]] #keep only the dates & voc conc. columns
        combined_df=first_df.rename(columns={voc:first_stationname}) #change the VOC column name to the station name
        first_present = True
    except:
        combined_df = first_df['Compounds']
    for (df,station_name) in df_pair_list[1:]: #for the other stations, merge their VOC conc. column with the first df
        try: #can only proceed if the VOC column is present for that station df
            voc_df = df[['Compounds',voc]].rename(columns={voc:station_name}) #rename column name to station name
            combined_df = combined_df.merge(voc_df, how='outer',
                                                on=['Compounds'])  # add a column with an 'outer' join
        except KeyError:
            pass
    combined_df.sort_values(by='Compounds', axis=0, ascending=True, inplace=True, ignore_index=True) #sort by time
    return combined_df

'''With modified ticks need to do'''
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
    fig, axes = plt.subplots(nrows=1,ncols=n_cols,figsize=(40,20))
    provinces = {1:'Newfoundland & Labrador',3:'Nova Scotia',4:'New Brunswick',5:'Quebec',6:'Ontario',7:'Manitoba',
                 8:'Saskatchewan',9:'Alberta',10:'British Columbia',12:'Nunavut'}
    i=0
    for (df,station) in province_data[province_int]: #unpack the df, station name tuples
        decades = [(1989, 1999), (1999, 2009), (2009, 2020)]  # get the start and end of each decade
        start = decades[0][0]
        end = decades[0][1]
        start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
        end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
        try: #do the following if the voc is present in the data for that station
            df_times = df[['Compounds',voc]][(df['Compounds'] >= start_yr) & (df['Compounds'] < end_yr)]
            df_times=df_times.rename(columns={voc:'1989\n1999'})
            #df_times=df_times.replace(to_replace=0,value=np.nan) '''Do not do anything to the zero values'''
            for start,end in decades[1:]:
                start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
                end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
                df_decade = df[['Compounds', voc]][(df['Compounds'] >= start_yr) & (df['Compounds'] < end_yr)]
                df_decade = df_decade.rename(columns={voc: '{}\n{}'.format(start,end)})
                #df_decade = df_decade.replace(to_replace=0, value=np.nan)
                df_times=df_times.merge(df_decade,how='outer',on=['Compounds'])
            df_times = df_times[['Compounds', '1989\n1999', '1999\n2009', '2009\n2020']]
            sns.boxplot(ax=axes[i],data=df_times,palette='Set2')
            font = {'family': 'serif',
                    'color': 'black',
                    'weight': 'normal',
                    'size': 14,
                    }
            axes[i].set_title('Station {}'.format(station),fontdict=font)
            ymax = find_ylim_boxplots2(df_times)
            axes[i].set_ylim(ymin=0, ymax= ymax)
            axes[i].set_ylabel('Concentration (ug/m3)', fontdict=font)
            axes[i].set_xlabel('Decades', fontdict=font)
            num_ticks = round(ymax/0.1)
            if num_ticks>46:
                interval = round((ymax/45),1)
                axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(interval))
                axes[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%{}f".format(interval)))
            else:
                axes[i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
                axes[i].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
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