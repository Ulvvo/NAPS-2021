import pandas as pd
import numpy as np
import os
from os import listdir
import json
import pickle
import time

'''General functions: making dataframes based on CSV files in their directories'''

'''For years 1989-2015. Note: for 2014 & 2015, there are two files for S62601 (4hr & 24hr).
This code takes the 4 hr files.'''


def get_paths(folderpath):
    # APPLICABLE FOR YEARS 1989-2015 ONLY
    '''Takes in a path of a year folder (eg. 2015) and returns a dictionary of the path links
    input: folderpath --> str --> "E://VOC Project/2010VOC/CSV"
    output: paths --> dict --> {str: str, str:str,...} --> {station1: filepath, station2: filepath, ...}
    '''
    # Applicable for years 1989 - 2015 only because they contain unique station files in their folders
    paths = {}  # create an empty dictionary
    with os.scandir(folderpath) as entries:  # produces a list of files inside directory
        for entry in entries:  # adds a key of filename and value of the filepath for each file
            if entry.is_file():
                paths[entry.name] = entry.path
    return paths


'''For years 2016-2019.Note: for 2016, there are two files for S62601 (4hr & 24hr).
This code takes the 4 hr files.'''


def get_paths2(folderpath):
    # APPLICABLE FOR YEARS 2016-2019 ONLY
    '''Takes in a path of a year folder (eg. 2015) and returns a dictionary of the path links
        input: folderpath --> str --> "E://VOC Project/2019VOC/VOC-COV"
        output: paths --> dict --> {str: str, str:str,...} --> {station1: filepath, station2: filepath, ...}
        '''
    # Applicable for years 2016 - 2019 only because they contain multiple files in English, French, Excel and CSV
    paths = {}  # create an empty dictionary
    filenames = listdir(folderpath)  # get a list of all files in the directory
    for filename in filenames:
        if filename.endswith('EN.csv'):
            paths[filename] = folderpath + '/{}'.format(filename)  # add the English CSV file to paths dictionary
    return paths


'''For all years'''


def skiprows(df):
    # APPLICABLE FOR ALL YEARS
    '''Finds the number of metadata rows to skip in order to get the proper column headings using an identifier word
    in the 1st column of the Dataframe. Identifier = 'Compounds' for years 1989-2013, 'Sampling Date' for years
    2014-2019.
    input: df (df) --> a raw version of Pandas DataFrame read directly from the CSV file
    eg. station 1, year 2010
    ---------------------------------------
    |   Metadata1                         |
    |   Metadata2                         |
    |   Metadata...                       |
    |   Compounds | VOC1  | VOC2 | ...    |
    |   1/1/2010  | 1.346 | 2.345| ...    |
    |   7/1/2010  | 3.4l5 | 2.489| ...    |
    |   13/1/2010 | 0.746 | 1.957| ...    |
    |    ...      |  ...  |  ... | ...    |
    |_____________________________________|

    output: skiplist (list) --> eg. [0,1,2] for 1st three rows to skip. If None, skip none. If False,
    leads to an Exception in one of the create_dfs functions.
    '''
    col1 = df.iloc[:, 0]  # get the first column of the raw dataframe
    # if the name of the first column itself is the identifier, then no need to skip any columns
    if (col1.name == 'Compounds') | (col1.name == 'Sampling Date'):
        skiplist = None  # None will not skip any columns
    else:
        # find the row returned where the identifier is (should return only 1)
        row = col1[(col1 == 'Compounds') | (col1 == 'Sampling Date')]
        if row.size == 0:  # if no rows returned, then return False
            skiplist = False
        else:
            ind = row.index.values[0]  # get the index value of the row containing the identifier
            skiplist = [i for i in range(0, ind + 1)]  # get a list of indices of rows to skip (from 0 until the index)
    return skiplist


'''For years 1989-2015'''


def create_dfs(paths):
    # APPLICABLE FOR YEARS 1989-2015 ONLY
    '''Takes in a dictionary of {filename:filepath}s and produces dataframes for every CSV file which is returned
    as a dictionary.

    input: paths --> for a given year (eg. 2010) {filename1:filepath1, filename2: filepath2, ...}
           skiprows (list) --> indices of rows to skip --> eg. [0,1]
    output: dfs --> {station1: df1, station2: df2...}

    The DataFrames produced in dfs have metadata rows & columns removed, as well as empty columns where no measurements
    were taken. *Some NaNs are still present in the data Type shown:
    eg. station 1, year 2010
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    '''
    dfs = {}  # create an empty dictionary
    for key in paths.keys():
        missingvals = ['-999', 'M1']  # add -999 and M1 codes to NaN values
        df_file = pd.read_csv(paths[key])  # skip 1st two lines of metadata
        rows = skiprows(df_file)  # find the indices of the rows to skip
        if rows == False:  # rows from skiprows() returns False if the identifier for the column head is not found
            raise Exception('COMPOUNDS not present in 1st col for station {}'.format(key[0:7]))
        del df_file  # delete the previous df_file to save space
        # read CSV again while removing metadata rows & converting the missing values to NaNs
        df_file = pd.read_csv(paths[key], skiprows=rows, na_values=missingvals)
        metadata = ['Sample ID#', 'Sample Date', 'Canister ID#', 'Sample Volume', 'NAPS ID',
                    'START TIME', 'DURATION']
        # remove any metadata columns that may be present in the dataframe
        labels = set(metadata).intersection(set(df_file.columns))
        df_file.drop(axis=1, labels=labels, inplace=True)  # drop metadata columns
        df_file.replace(' ', np.nan, inplace=True)  # ensure the blank spaces also read as NaNs
        df_file.dropna(axis=1, how='all', inplace=True)  # removes columns made up entirely of NaNs
        col1name = df_file.columns[0]  # get the first column's name
        try:
            col1 = df_file[col1name]
            # change the dates to Pandas TimeFrame format
            df_file[col1name] = pd.to_datetime(col1, utc=False, infer_datetime_format=True)
            dfs[key[0:7]] = df_file  # add key as station name and df as the value
            # only add the file that allows the conversion to datetime
        except:  # ignore the random files with transposed columns (eg. S62601_VOCS in year 2006)
            pass
    return dfs


'''For years 2016-2017'''


def create_dfs2(paths):
    # APPLICABLE FOR YEARS 2016-2017 ONLY
    '''Takes in a dictionary of {filename:filepath}s and produces dataframes for every CSV file which is returned
    as a dictionary.

    input: paths --> for a given year (eg. 2016) {filename1:filepath1, filename2: filepath2, ...}
           skiprows (list) --> indices of rows to skip --> eg. [0,1]
    output: dfs --> {station1: df1, station2: df2...}

    The DataFrames produced in dfs have metadata rows & columns removed, as well as empty columns where no measurements
    were taken. *Some NaNs are still present in the data Type shown:
    eg. station 1, year 2016
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    '''
    dfs = {}  # create an empty dictionary
    for key in paths.keys():
        missingvals = ['-999', 'M1']  # add -999 and M1 codes to NaN values
        rows = [0, 1, 2]  # for 2016 & 2017, the first three rows are metadata
        df_file = pd.read_csv(paths[key], skiprows=rows, na_values=missingvals)
        metadata = ['Sample ID#', 'Sample Date', 'Canister ID#', 'Sample Volume', 'NAPS ID',
                    'START TIME', 'DURATION']
        # remove any metadata columns that may be present in the dataframe
        labels = set(metadata).intersection(set(df_file.columns))
        df_file.drop(axis=1, labels=labels, inplace=True)  # drop metadata columns
        indicator = 'File generated on'
        rows_bottom = df_file[df_file['Sampling Date'] == indicator]  # identify the bottom metdata 1st row
        ind = rows_bottom.index.values[0]  # get the index of the indicator row
        drop = [i for i in range(ind - 2, len(df_file))]  # delete all rows below the indicator & 2 rows above
        df_file.drop(axis=0, index=drop, inplace=True)
        df_file.replace(' ', np.nan, inplace=True)  # replace all blanks with NaNs
        df_file.dropna(axis=1, how='all', inplace=True)  # removes columns made up entirely of NaNs
        col1name = df_file.columns[0]  # get the first column's name
        try:
            col1 = df_file[col1name]
            # change the dates to Pandas TimeFrame format
            df_file[col1name] = pd.to_datetime(col1, utc=False, infer_datetime_format=True)
            dfs[key[0:7]] = df_file  # add key as station name and df as the value
        except:  # ignore the random files with transposed columns (eg. S62601_VOCS in year 2006)
            pass
    return dfs


'''For years 2018-2019'''


def create_dfs3(paths):
    # APPLICABLE FOR YEARS 2018-2019 ONLY
    '''Takes in a dictionary of {filename:filepath}s and produces dataframes for every CSV file which is returned
    as a dictionary.

    input: paths --> for a given year (eg. 2018) {filename1:filepath1, filename2: filepath2, ...}
           skiprows (list) --> indices of rows to skip --> eg. [0,1]
    output: dfs --> {station1: df1, station2: df2...}

    The DataFrames produced in dfs have metadata rows & columns removed, as well as empty columns where no measurements
    were taken. *Some NaNs are still present in the data Type shown:
    eg. station 1, year 2018
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    '''
    dfs = {}  # create an empty dictionary
    for key in paths.keys():
        missingvals = ['-999', 'M1']  # add -999 and M1 codes to NaN values
        rows = [i for i in range(0, 72)]  # at least 72 rows of metadata are present (estimate of rows to skip)
        colskeep = [i * 3 for i in range(1, 110)]  # each VOC has 3 columns so this only keeps the 1st col for each
        colskeep.append(1)  # includes the 'Sampling Date' column
        # use encoding "ISO-8859-1" for files in 2018/2019 because the usual "utf-8" cannot decipher code
        df_file = pd.read_csv(paths[key], skiprows=rows, usecols=colskeep, na_values=missingvals, encoding="ISO-8859-1")
        rowsskip = skiprows(df_file)  # find exact number of rows to skip
        if rowsskip != None:
            try:
                rows = [i for i in range(0, 72 + len(rowsskip))]  # adjust the number of rows to skip
                # read the CSV file again
                df_file = pd.read_csv(paths[key], skiprows=rows, usecols=colskeep, na_values=missingvals,
                                      encoding="ISO-8859-1")
            except TypeError:  # Error returned when rowsskip is False
                raise Exception('Ethylene not found in 1st column of station {}'.format(key[0:7]))
        metadata = ['Sample ID#', 'Sample Date', 'Canister ID#', 'Sample Volume', 'NAPS ID',
                    'START TIME', 'DURATION']
        labels = set(metadata).intersection(set(df_file.columns))
        df_file.drop(axis=1, labels=labels, inplace=True)  # drop metadata columns
        df_file.replace(' ', np.nan, inplace=True)  # replace blanks with NaNs
        df_file.dropna(axis=1, how='all', inplace=True)  # removes columns made up entirely of NaNs
        col1name = df_file.columns[0]  # get the first column's name
        try:
            col1 = df_file[col1name]
            # change the dates to Pandas TimeFrame format
            df_file[col1name] = pd.to_datetime(col1, utc=False, infer_datetime_format=True)
            dfs[key[0:7]] = df_file  # add key as station name and df as the value
        except:  # ignore the random files with transposed columns (eg. S62601_VOCS in year 2006)
            pass
    return dfs


'''This function creates a final_dfs dictionary containing the entire set of data over all years, stations and VOCs'''


def final_dfs():
    '''Compiles all data for all years, stations and VOCs into a nested dictionary.
    input: None
    output: final_dfs (nested dict) of format:
    final_dfs = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }
    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |
    '''
    final_dfs = {}
    for year in range(1989, 2016):  # range 1989-2016
        folderpath1 = "E://VOC Project/{}VOC/CSV".format(year)  # get path of folder
        paths1 = get_paths(folderpath1)  # get a dictionary of file paths
        dfs1 = create_dfs(paths1)  # create a dict of dfs for all the station files in that year folder
        final_dfs[year] = dfs1  # create a nested dict entry for that year with the value as the dict of dfs
        print(year)  # print the year to show progress
    for year in range(2016, 2018):
        folderpath2 = "E://VOC Project/{}VOC/VOC-COV".format(year)
        paths2 = get_paths2(folderpath2)
        dfs2 = create_dfs2(paths2)
        final_dfs[year] = dfs2
        print(year)
    for year in range(2018, 2020):
        folderpath3 = "E://VOC Project/{}VOC/VOC-COV".format(year)
        paths3 = get_paths2(folderpath3)
        dfs3 = create_dfs3(paths3)
        final_dfs[year] = dfs3
        print(year)
    return final_dfs


'''This function is needed to format the entire collection of dataframes (final_dfs) in a standard way'''


def format_all(alldata):
    '''Takes in a nested dictionary containing data on VOCs, years & stations and produces another dictionary
    with all the station & column names standardized. All 5-digit station names will be converted from having a '_' at
    the end (eg. 'S52601_') to having a '0' in front (eg. 'S052601'). All column names containing '(ug/m3)'
    after the VOC name (eg. 'Ethane (ug/m3)') will have it removed (eg. 'Ethane').

    input: alldata (nested dict) of format:
    alldata = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }
    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    output: formatted data (nested dict) --> alldata with formatting changes
    '''
    formatted_data = {}  # create a new empty dictionary for formatted data
    for year in alldata:
        if year in range(1989, 2014):
            formatted_data[year] = {}
            for station in alldata[year]:
                if station[-1] == '_':
                    new_name = station[0] + '0' + station[1:6]  # modify name with a '0' in front
                    formatted_data[year][new_name] = alldata[year][station]  # add the station data with a new name
                else:
                    formatted_data[year][station] = alldata[year][station]  # otherwise add the old name
        elif year in range(2014, 2020):
            formatted_data[year] = {}
            for station in alldata[year]:
                df = alldata[year][station]
                # cut out '(ug/m3)' from all columns names
                new_cols = df.columns.str.replace(' (ug/m3)', '', regex=False)
                # Propane & Propylene have a '.1' after their names (eg. 'Propane.1') so remove that
                new_cols = new_cols.str.replace('.1', '', regex=False)
                # replace the 'Sampling Date' identifier with 'Compounds'
                new_cols = new_cols.str.replace('Sampling Date', 'Compounds', regex=False)
                df.columns = new_cols  # change df column names to the formatted ones
                if station[-1] == '_':  # S62601_ is the only station with a '_' at the end so modify that
                    new_name = station[0] + '0' + station[1:6]
                    formatted_data[year][new_name] = df  # add the df with a new name as key
                else:
                    formatted_data[year][station] = df  # for all the other stations, keep the same name
    return formatted_data


'''Note: this function collapse_years() simply collapses the years onto the time dimension. The input to this HAS to be
FORMATTED alldata (eg. from format_all()), otherwise the combined dataframe will have lots of NaTs in the 1st col.

It does not account for any repeats of VOC columns due to changes in spelling, etc. 
Eg. There might be two different columns present:

"Freon11" & "Freon 11" even though they are the same VOC

That is where repeat_cols() and merge_repeated() functions come in to eliminate repeats.
'''


def collapse_years(alldata):
    '''Takes in a nested dictionary containing data on VOCs, years & stations & produces another
    dictionary with the years dimension collapsed onto 'Time'.

    input: alldata (nested dict) of format:
    alldata = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }
    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    output: collapsed_data (dict) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |
    '''
    bunched_stations = {}  # first create an empty dict
    # values will be lists with the same station dataframes but at different years:
    # eg. bunched_stations = {station1:[year1df,year2df,...], station 2:[...],...}
    for year in alldata:
        for station in alldata[year]:
            df_station = alldata[year][station]  # this is the df of a given station at a given year
            try:
                bunched_stations[station].append(df_station)  # if the station already exists then append to the list
            except KeyError:
                bunched_stations[station] = [df_station]  # if the station doesn't exist, create a new entry
    collapsed_years = {}  # create an empty dictionary for the collapsed year dfs
    for station in bunched_stations:
        dfs_years = bunched_stations[station]  # get the list of dfs for the same station over the years
        combined_df = pd.concat(dfs_years,
                                ignore_index=True)  # concatenate the dfs into a single df spanning several years
        collapsed_years[station] = combined_df  # add the combined df into the new dict
    return collapsed_years


'''This function collapses the years in a given range only'''


def collapse_years2(alldata, start, end):
    '''Takes in a nested dictionary containing data on VOCs, years & stations, and a start and end year
    & produces another dictionary with the years dimension collapsed onto 'Time'.

    input: start (int) --> start year of data to collapse
            end (int) --> end year of data to collapse
            alldata (nested dict) --> nested dict containing all data of format:
    alldata = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }
    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    output: collapsed_data (dict) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years <start> - <end>) eg. 1989 - 2011, or 1989-2019, or 2010-2013
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |
    '''
    bunched_stations = {}  # first create an empty dict
    # values will be lists with the same station dataframes but at different years:
    # eg. bunched_stations = {station1:[year1df,year2df,...], station 2:[...],...}
    for year in range(start, end):
        for station in alldata[year]:
            df_station = alldata[year][station]  # this is the df of a given station at a given year
            try:
                bunched_stations[station].append(df_station)  # if the station already exists then append to the list
            except KeyError:
                bunched_stations[station] = [df_station]  # if the station doesn't exist, create a new entry
    collapsed_years = {}  # create an empty dictionary for the collapsed year dfs
    for station in bunched_stations:
        dfs_years = bunched_stations[station]  # get the list of dfs for the same station over the years
        combined_df = pd.concat(dfs_years,
                                ignore_index=True)  # concatenate the dfs into a single df spanning several years
        collapsed_years[station] = combined_df  # add the combined df into the new dict
    return collapsed_years

def combine_stations(stations_to_combine,alldata):
    '''
    Combines the Dataframes of multiple stations into singular DataFrames. To be used when multiple stations are
    basically the 'same' station  - i.e. in the same area & alternated over the course of decades.

    Takes in a list of tuples containing station IDs & a dictionary of DataFrames containing all the collapsed data for
    all stations. Returns the same dictionary but with the list of stations combined into single DataFrames.

    inputs:
    stations_to combine (list of tuples) --> all the stations in each tuple will be combined together. The first element
    is the dataframe for that station and the last element in the tuple will be the name of the final combined DataFrame.
    eg. [(s1,s2,s3,s2),(s5,s6,s7,s5),(s9,s10,s11,s11)]
    *here, the stations s1,s2,s3 will be combined together and named s2, s5,s6,s7 will be combined together and named
    s5,etc.

    alldata (nested dict of dfs) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019) or whichever is the start and end time for observations
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    output: alldata but with stations in tuples combined into merged DataFrames.
    '''
    new_all_data = {key:value for key,value in alldata.items()}
    for stations in stations_to_combine: #iterate over the list of lists
        station_name = stations[0] #get the first element of the tuple - this is the station name to be given
        dfs_list = [alldata[station] for station in stations]
        combined_df = pd.concat(dfs_list,ignore_index=True)
        combined_df.sort_values(by='Compounds', axis=0, ascending=True, inplace=True, ignore_index=True)
        new_all_data[station_name]=combined_df
        removed_dfs = [new_all_data.pop(station) for station in stations[1:]]
    return new_all_data

def group_by_province(alldata):
    '''
    Takes in a dictionary of {station:df,...} and groups stations by their province represented by NAPS station codes.

    input:alldata (dict of dfs) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019) or whichever is the start and end time for observations
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    output: province_data (dict) --> a dictionary with keys being NAPS station beginning codes & values as list of
    tuples of station Dfs & station names.

    eg. province_data = {1: [(df10102,'S010102')], 3: [(df30112,'S030113'),(df30118,'S030118'),...],
    4: [(df40203,'S040203'),...],5: [(df50115,'S050115'),...], 6: [...], 7: [...], 8: [...], 9: [...], 10: [...]}

    Legend for NAPS province codes:
    1 --> station numbers beginning with 1 (eg. "S010102") --> represents Newfoundland & Labrador
    3 --> station numbers beginning with 3 (eg. "S030113") --> represents Nova Scotia
    4 --> station numbers beginning with 4 (eg. "S040203") --> represents New Brunswick
    5 --> station numbers beginning with 5 (eg. "S050115") --> represents Quebec
    6 --> station numbers beginning with 6 (eg. "S060101") --> represents Ontario
    7 --> station numbers beginning with 7 (eg. "S070119") --> represents Manitoba
    8 --> station numbers beginning with 8 (eg. "S080901") --> represents Saskatchewan
    9 --> station numbers beginning with 9 (eg. "S090227") --> represents Alberta
    10 --> station numbers beginning with 10 (eg. "S100110") --> represents British Columbia
    12 --> station numbers beginning with 12 (eg. "S129401") --> represents Nunavut
    '''
    province_data = {1: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [],12:[]}
    for station in alldata:
        if station[0:2] != 'S1':
            province_data[int(station[2])].append((alldata[station], station))
        elif station[0:3]=='S10':
            province_data[10].append((alldata[station], station))
        elif station[0:3]=='S12':
            province_data[12].append((alldata[station], station))
    return province_data

def check_province_voc(province,voc,province_data):
    '''Returns a Boolean based on whether the given VOC is present in at least one of the dfs of the stations in
    province data.'''
    df_pair_list = province_data[province]
    voc_present = False
    stations_voc_present = [] #create an empty list containing names of stations with the given VOC present
    for (df, station_name) in df_pair_list:
        try:
            voc_df = df[voc]
            voc_present = True
            stations_voc_present.append(station_name)
        except KeyError:
            pass
    #returns a boolean of whether the VOC is present in any of the stations
    try:
        first_station = stations_voc_present[0]
    except IndexError:
        first_station = None
    return voc_present,first_station

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
    #gives a True/False whether the VOC is present in any of the Dataframes of the stations for this province
    (voc_present,first_station) = check_province_voc(province,voc,province_data)
    if voc_present==True: #only if the VOC is present in any one of the stations within the province
        df_pair_list = province_data[province] #get the list of (df, station_name) tuples for stations in that province
        station_names = [name for (df,name) in df_pair_list] #just get the station names
        first_station_index = station_names.index(first_station) #get the index of the first station containing the voc
        #the combined df starts off with the df of the first station containing the VOC
        combined_df = df_pair_list[first_station_index][0][['Compounds',voc]].rename(columns={voc:first_station})
        #for all the stations after the starting station, try get the column of VOC & merge it with the 'combined_df'
        for (df,station_name) in df_pair_list[(first_station_index+1):]:
            try:  # can only proceed if the VOC column is present for that station df
                voc_df = df[['Compounds', voc]].rename(columns={voc: station_name})  # rename column name to station name
                combined_df = combined_df.merge(voc_df, how='outer', on=['Compounds'])  # add a column with an 'outer' join
            except KeyError:
                pass #if the next station does not have the VOC in its columns, do nothing
        combined_df.sort_values(by='Compounds', axis=0, ascending=True, inplace=True, ignore_index=True) # sort by time
        return combined_df
    else:
        return None #if VOC is not present in ANY of the stations, then return None

def select_10(alldata):
    '''
    Returns the alldata dict but only containing stations with 10 yrs+ of data.
    input: alldata (dict of dfs) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019) or whichever is the start and end time for observations
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    output: alldata but only containing stations with 10 yrs+ data
    '''
    new_dict = {}
    for station in alldata:
        df = alldata[station]
        start = df['Compounds'].iloc[0].year
        end = df['Compounds'].iloc[-1].year
        years = end-start
        if years >= 10:
            new_dict[station]=df
    return new_dict

'''These functions work on exporting/importing files'''
## PICKLE THIS DICTIONARY TO A SERIALIZED FORMAT
'''Pickling usually takes ... ~ 6 seconds for loading dict when opening. ~0.8 seconds when re-loading'''


def export_pickle(final_dfs, filepath):
    '''Exports the final_dfs dictionary as a Pickle file'''
    with open(filepath, 'wb') as pickle_out:  # open file
        pickle.dump(final_dfs, pickle_out)  # dump Pickle into file
        pickle_out.close()  # close file


def import_pickle(filepath):
    '''Takes a pickle and converts it to its Python format & also returns the time taken to execute
    the function.
    '''
    start_time = time.time()  # takes time at the start of the function
    with open(filepath, 'rb') as pickle_in:  # open file in read mode
        alldata = pickle.load(pickle_in)  # load pickle file
        pickle_in.close()  # close file
    time_taken = time.time() - start_time  # find the time difference
    return alldata, time_taken

'''JSON usually takes ~100 seconds for loading first time & then 60 seconds for re-loading'''

def export_json_dict(final_dfs, filepath):
    '''
    Takes in a dictionary of all the data & exports it as a JSON file
    input:
    filepath (str) --> eg."E://User/folder1/folder2/sample_json_file.JSON"
    final_dfs (nested dict) --> eg.

    final_dfs = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }
    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |

    output:
    file (JSON) of format {year1:{station1: {JSON DICT OF DATAFRAME}, station2:{JSON DICT OF DATAFRAME},...},...}
    '''
    new_dict = {}  # create a new dictionary
    # first convert all the DataFrames of each year & station to JSON objects
    for year in final_dfs:
        new_dict[year] = {}  # create an empty nested dictionary for each year as a key in new_dict
        for station in final_dfs[year]:
            df = final_dfs[year][station]  # access the station dataframe
            json_object = df.to_json(orient='table')  # convert the dataframe to a JSON object with 'table' formatting
            new_dict[year][station] = json_object  # add JSON object into the nested empty dict with the station as key
    with open(filepath, 'w') as outfile:  # open file
        json.dump(new_dict, outfile)  # dump dictionary as JSON contents into file
        outfile.close()  # close file


def import_json_dict(filepath):
    '''
    Takes in a filepath of a JSON file & then converts it to a nested dictionary full of dfs. Returns two objects: 1
    nested dictionary, and the time taken for the function to complete.

    input: filepath (str) --> eg."E://User/folder1/folder2/sample_json_file.JSON"
    where the JSON file contains data in the format:
    {year1:{station1: {JSON DICT OF DATAFRAME}, station2:{JSON DICT OF DATAFRAME},...},...}

    output: final_dfs (nested dict) --> eg.
    final_dfs = {year1:{station1:df1,station2:df2,...station(i):df(i)}, year2:{station1:df1,station2:df2,...
    station(i):df(i)},...,year(j):{station1:df1,station2:df2,...station(i):df(i)} }

    where each df is of the format:
    eg. year j, station i: (only one year at one station)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/2010  | 1.346 | 2.345| ... |
    |   7/1/2010  | 3.4l5 | 2.489| ... |
    |   13/1/2010 | 0.746 | 1.957| ... |
    |    ...      |  ...  |  ... | ... |
    '''
    start_time = time.time()  # takes starting time
    alldata = {}  # create an empty dict
    with open(filepath, 'r') as infile:  # open file
        json_dict = json.load(infile)  # load JSON dict contents
    for year in json_dict:  # access the value of each year key of the JSON dict
        alldata[int(year)] = {}  # create an empty dict (nested) with year as key
        for station in json_dict[year]:
            json_object = json_dict[year][station]  # get the JSON formatted dataframe from the value of the station key
            df = pd.read_json(json_object, orient='table')  # convert dict to Pandas DataFrame object
            alldata[int(year)][
                station] = df  # add dataframe to the alldata dict, as the value for the station key (nested)
    time_taken = time.time() - start_time  # get the time difference
    return alldata, time_taken


def export_json_dfs(df, filepath):
    '''
    Exports a Pandas DataFrame to a JSON file
    input: df (Pandas DataFrame)
           filepath (str) --> eg."E://User/folder1/folder2/sample_json_file.JSON"
    output: JSON file written in specified path location
    '''
    df.to_json(filepath)


def repeatcols(coll_data, allvocs):
    '''
    Finds the difference between the VOCs present in the collapsed DataFrame for each station and the complete list
    of VOCs. Helps identify excess/repeat columns in the collapsed DF due to mispelling, etc.
    *Requires data to be collapsed (years dimension on time - i.e. time measured in dates over
    years) using prior functions like collapse_years() or collapse_years2().

    Input: coll_data (dict) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019/start-end)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    allvocs (Pandas Series) --> eg. ['VOC1' 'VOC2' 'VOC3' ...]. Imported from 'allvocs.pickle' using import_pickle()

    Output: diff (dict) --> dict containing stations as keys and sets of VOC names as values. These VOCs are the ones
    repeated in the DF due to mis-spelling, etc.
    eg. {station1:{'VOC1','VOC2'},station2:{}, station3: {'VOC5','VOC9'},...}
    '''
    s1 = set(allvocs)  # get the set of the complete VOCs list
    diff = {}  # create an empty dictionary
    for station in coll_data:
        df = coll_data[station]  # get the collapsed df for each station in the coll_data dictionary
        s2 = set(df.columns.tolist())  # turn the column names of the station df into a set
        difference = s2.difference(s1)  # find the difference between the station VOCs & the complete set of VOCs
        difference.remove('Compounds')  # remove 'Compounds' from the set because it is not a VOC
        diff[station] = difference  # add the difference set to the diff dict with 'station' as the key
    return diff


def newcols(coll_data, allvocs):
    '''
    Finds the difference between the complete list of VOCs and the VOCs present in the collapsed DataFrame for each
    station. Helps identify VOCs that were simply not measured for a given station over its entire history.
    *Requires data to be collapsed (years dimension on time - i.e. time measured in dates over
    years) using prior functions like collapse_years() or collapse_years2().

    Input: coll_data (dict) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019/start-end)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    allvocs (Pandas Series) --> eg. ['VOC1' 'VOC2' 'VOC3' ...]. Imported from 'allvocs.pickle' using import_pickle()

    Output: diff (dict) --> dict containing stations as keys and sets of VOC names as values. These VOCs are the ones
    never measured for a given station over its entire period of measurements.
    eg. {station1:{'VOC1','VOC2'},station2:{}, station3: {'VOC5','VOC9'},...}
    '''
    s1 = set(allvocs)  # get the set of the complete VOCs list
    diff = {}  # create an empty dictionary
    for station in coll_data:
        df = coll_data[station]  # get the collapsed df for each station in the coll_data dictionary
        s2 = set(df.columns.tolist())  # turn the column names of the station df into a set
        s2.remove('Compounds')  # remove 'Compounds' from the set because it is not a VOC
        difference = s1.difference(s2)  # find the difference between the complete set of VOCs & the station VOCs
        diff[station] = difference  # add the difference set to the diff dict with 'station' as the key
    return diff


def merge_repeated(coll_data, diff):
    '''
    Takes a dictionary containing collapsed VOC dataframes for all stations (i.e. over all years), and a dictionary of
    repeated VOCs and then modifies the columns of each dataframe accordingly to eliminate repeated VOC columns due to
    mis-spelling. Eg. Combines columns like 'Freon11' and 'Freon 11'  which were previously interpreted as different
    due to the space between the words.

    input: coll_data (dict) --> {station1:df1, station2:df2, ..., station(i):df(i)},
    where each df is of the format:
    eg. station1: (time goes from years 1989-2019/start-end)
    |   Compounds | VOC1  | VOC2 | ... |
    ----------------------------------------
    |   1/1/1989  | 1.346 | 2.345| ... |
    |    ...      |  ...  |  ... | ... |
    |    ...      |  ...  |  ... | ... |
    |  31/12/2019 | 2.669 | 5.876| ... |

    diff (dict) --> dict containing stations as keys and sets of VOC names as values. These VOCs are the ones
    repeated in the DF due to mis-spelling, etc.
    eg. {station1:{'VOC1','VOC2'},station2:{}, station3: {'VOC5','VOC9'},...}

    output: coll_data but with repeated VOC columns combined into single ones. More details below.

    Specific changes:
    All the 'Freons': Keep 'FreonX' from 1989-2013, and combine it with 'Freon X' from 2014-2019. Then delete 'Freon X'.
    Results in a single column called 'FreonX' from 1989-2019. *Note: X = 11,12,22,114

    All the monoterpenes: Keep 'x-Monoterpene' from 1989-2017 and combine it with 'X-Monoterpene' from 2018-2019. Then
    delete 'X-Monoterpene'. Results in a single column called 'x-Monoterpene' from 1989-2019. *Note: x = a,b,d and
    X = A,B,D and Monoterpene = Pinene, Limonene

    All the methyl-alkenes (eg. 3-Methyl-1-butene, cis-3-Methyl-2-pentene, etc.): Keep the lowercase version of the
    alkene from 1989-2017 (eg. -pentene, -butene) and combine it with the uppercase version from 2018-2019 (eg.
    -Pentene, -Butene). Results in single columns of the lowercase versions from 1989-2019.

    iso-Propylbenzene: Keep 'iso-Propylbenzene' from 1989-2017 and combine it with 'Iso-Propylbenzene' from 2018-2019.
    Results in a single lowercase column 'iso-Propylbenzene' from 1989-2019.

    1-Hexene: Keep '1-Hexene' from 1989-2013 and combine it with '1-Hexene/2-Methyl-2-Pentene' from 2014-2019. Results
    in a single column '1-Hexene' from 1989-2019.
    '''
    for station in diff:
        df = coll_data[station]
        set_VOCs = diff[station]
        for voc in set_VOCs:
            if voc[0:5] == 'Freon':  # identify the VOC
                if voc[-3:] == '114':  # if number is 114, then new_name has a different slicing
                    # the new name is technically the older VOC name, but we assign the later mis-spelled column this
                    new_name = 'Freon' + voc[-3:]
                else:
                    new_name = 'Freon' + voc[-2:]  # otherwise, create a new name to eliminate the blank space between
                    # get the year start for which the VOC name changed from new_name to the mis-spelled one
                years = pd.Timestamp('2014-01-01')
            if voc[-4:] == 'nene':
                new_name = voc[0].lower() + voc[1:]
                years = pd.Timestamp('2018-01-01')
            if voc[-6:] == 'Butene':
                new_name = voc[0:-6] + 'butene'
                years = pd.Timestamp('2018-01-01')
            if voc[-7:] == 'Pentene':
                new_name = voc[0:-7] + 'pentene'
                years = pd.Timestamp('2018-01-01')
            if voc[0:3] == 'Iso':
                new_name = 'iso' + voc[3:]
                years = pd.Timestamp('2018-01-01')
            if voc[0:8] == '1-Hexene':
                new_name = '1-Hexene'
                years = pd.Timestamp('2014-01-01')
            try:  # if the new_name is present already in the DF as its own column,
                # assign its values for the year switch onwards to the values of the mis-spelled column
                df[new_name][df['Compounds'] >= years] = df[voc][df['Compounds'] >= years]
                df.drop(labels=[voc], axis=1, inplace=True)  # delete the mis-spelled column afterwards
            except KeyError:  # if the new_name doesn't exist, then create a new column with the VOC correctly spelled
                ind = df.columns.get_loc(voc)
                df.insert(ind, new_name, df[voc])
                df.drop(labels=[voc], axis=1, inplace=True)
    return coll_data

def import_selected(filepath,stationlist):
    '''
    Imports a selected bunch of stations from the collapsed all_data dictionary of dataframes.
    input: filepath (str) --> path to the collapsed all_data pickle file

    *the all_data pickle file is a dictionary of the form:
    {station1:{'VOC1','VOC2'},station2:{}, station3: {'VOC5','VOC9'},...}

           stationlist (list) --> list of stations to import from all_data

    output: selected_data (dict) --> a dictionary containing selected station names as keys and dataframes as values
    of the form:
    {station1:{'VOC1','VOC2'},station2:{}, station3: {'VOC5','VOC9'},...}
    '''
    all_data = import_pickle(filepath)[0] #import the dictionary containing all the data
    # create a new dictionary with selected stations
    selected_data = {station:all_data[station] for station in stationlist}
    del all_data
    return selected_data



alldata = import_pickle('E://Summer research/pythonProject/data/colldatav4.pickle')[0]
# #stations = ['S030113','S030117','S030118']
# # myd = {}
# # for station in stations:
# #     myd[station]=alldata[station]
#
# stations = [('S030118','S030113','S030117'),('S050104','S050134'),('S050115','S050136'),('S050121','S050122'),
#             ('S060403','S060429','S060435'),('S060903','S060904'),('S061004','S061009'),('S063201','S065101'),
#             ('S080110','S080111'),('S090228','S090227','S090230'),('S101004','S101005')]
#
# sel = select_10(alldata)
