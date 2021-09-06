import pandas as pd
import numpy as np
from DataCleaning import get_paths, get_paths2, skiprows,create_dfs, create_dfs2, create_dfs3

'''VOC-related functions'''

'''Generate a list of all possible VOCs measured at different stations over different years'''
def vocs_year(dfs):
    '''Takes in a dictionary of dataframes {station1:df1, station2:df2,...} for different stations for a given year
    folder and returns a dataframe (df) with each column containing VOCs measured at a given station.
    input: paths (dict) --> {station1: df1, station2: df2,...}
           where df1, df2,... are of the form: eg. station1, year 2010
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    output: df (df) --> eg. for year 2010:
      | Station1 |  Station2 |... | Station(n-1) | Station(n)
      -------------------------------------------------------
    0 |  VOC1    |   VOC1    |... | VOC1         | VOC1
    1 |  VOC2    |   VOC2    |... | VOC2         | VOC2
    2 |  ...     |   ....    |... | ....         | ....
    '''
    #create an empty list
    dflist = []
    for station in dfs.keys():
        df_file = dfs[station] #access the dataframe
        vocs = df_file.columns[1:] #after the 1st col, all the column names should be VOCs
        dfcols = pd.DataFrame(vocs) #create a 1D dataframe made up entirely of a list of vocs
        dfcols.columns=[station] #assign the column name as the station name
        del df_file  # delete df to reduce RAM usage
        dflist.append(dfcols) #add the dfcols DF to an empty list
    df = pd.concat(dflist,axis=1) #concatenate all the DFs present in the list together
    df.drop(axis=0,index=0,inplace=True)
    return df

def voc_year_union(df):
    '''
    Takes a dictionary of stations and dataframes Dataframe containing all VOCs at different stations and produces a union DataFrame for that year
    input: df (df), eg. for year 2010:
      | Station1 |  Station2 |... | Station(n-1) | Station(n)
      -------------------------------------------------------
    0 |  VOC1    |   VOC1    |... | VOC1         | VOC1
    1 |  VOC2    |   VOC2    |... | VOC2         | VOC2
    2 |  ...     |   ....    |... | ....         | ....

    output: dfnew (df), eg. at year 2010:
      | VOCs
    ----------
    0 | VOC1
    1 | VOC2
    2 | ....
    n | VOC(n)
        '''
    # first change all column names of the input df to "VOCs"
    for column in df.columns:
        column='VOCs'
    # now for every pair of columns, merge adjacent cols to produce a union
    for i in range(0, len(df.columns) - 1):
        df_col1 = df.iloc[:, i] #the first column of df as its own dataframe
        df_col2 = df.iloc[:, i + 1] #the adjacent col. of df as its own dataframe
        dfnew = pd.merge(df_col1, df_col2, 'outer') #choose outer merge so that the union is generated
    return dfnew

def overall_voc_union(dflist):
    '''
    Takes a list of dataframes with VOC compound unions for each year and then
    produces a union DataFrame of all VOCs over the years

    input: dflist (list) --> [df_year1, df_year2, df_year3, ...]
    where each df is of the type: eg. all stations, year 2010
      | VOCs
    ----------
    0 | VOC1
    1 | VOC2
    2 | ....
    n | VOC(n)

    output: dfnew (df), for all years and stations
      | VOCs
    ----------
    0 | VOC1
    1 | VOC2
    2 | ....
    n | VOC(n)
    '''
    for i in range(0,len(dflist)-1):
        df1 = dflist[i] #merge adjacent columns of the df to create a union
        df2 = dflist[i+1]
        dfnew = pd.merge(df1,df2,'outer')
    return dfnew

def find_all_vocs(start, end):
    '''
    Finds all possible VOCs measured for all the years (1989 - 2015)
    input: start (int) & end (int) --> eg. 1989
    output: df (df), all stations, all years
      | VOCs
    ----------
    0 | VOC1
    1 | VOC2
    2 | ....
    n | VOC(n)
    '''
    dflist = []
    for year in range(start,end):
        folderpath = "E://VOC Project/{}VOC/CSV".format(year)
        if year in range(1989,2016):
            path = get_paths(folderpath)
            dfs = create_dfs(path)
        elif year in range(2016,2018):
            path = get_paths2(folderpath)
            dfs = create_dfs2(path)
        elif year in range(2018,2020):
            path = get_paths(folderpath)
            dfs=create_dfs3(path)
        df_year = vocs_year(dfs)
        df_year_union = voc_year_union(df_year)
        dflist.append(df_year_union)
    voc_df = overall_voc_union(dflist)
    return voc_df

'''Compares the VOCs measured at different stations (and different years) with the entire list of possible VOCs'''
def VOClist(allvocs, dfs):
    '''Takes in a dictionary of Dataframes for different stations of a given year & an array of all VOCs
    and outputs an excel file of 0 and 1s to indicate the absence/presence (respectively) of each particular VOC.

    input: allvocs (np array) --> [VOC1 VOC2 ...]
           dfs (dict) --> {station1:df1, station2: df2, ...}
           where each df is of the type: eg. for station 1, year 2010
                        VOC1    VOC2    ...
            1/1/2010    1.346   2.345   ...
            7/1/2010    3.4l5   NaN     ...
            13/1/2010   0.746   1.957   ...
            ...         ...     ...     ...
    output: Excel written from a dataframe
    where the table is of the type: eg. for year 2010
                        Station1   Station2   ...
            VOC1        1           1         ...
            VOC2        0           1         ...
            VOC3        1           1         ...
            ...         ...         ...       ...
    '''
    df1= pd.DataFrame()
    for station in dfs:
        df = dfs[station]
        arr = []
        for voc in allvocs:
            '''Replace voc with (voc+' (ug/m3)' if year is 2014-2017.'''
            if voc in df.columns:
                arr.append(1)
            else:
                arr.append(0)
        arr = np.array(arr)
        df1[station]=arr
    return df1

'''Running zone'''
# allvocs = pd.read_excel("E://VOC Project/allvocs.xlsx")
# allvocs = allvocs['Compounds'].to_numpy()

'''Finding VOCs for all years'''
# for year in range(2014,2015):
#     folderpath = "E://VOC Project/{}VOC/CSV".format(year)
#     paths = get_paths(folderpath)
#     dfs = create_dfs(paths)
#     vy=vocs_year(dfs)
#     vl = VOClist(allvocs,dfs)
#     with pd.ExcelWriter("E:\VOC Project\output.xlsx",mode='a') as writer:
#         vl.to_excel(writer,sheet_name='{}'.format(year))
#         writer.save()
#     print(year)
