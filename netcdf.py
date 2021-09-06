import pandas as pd
import numpy as np
from DataCleaning import get_paths, get_paths2, skiprows,create_dfs, create_dfs2, create_dfs3, import_pickle, export_pickle
import pickle
import netCDF4
# from cftime import date2num, num2date, datetime

'''In this document, prepare data & create NETCDF file'''

'''Get data for all years, stations & VOCs from serialized Pickle object'''
alldata = import_pickle("E://Summer research/pythonProject/data/colldata_uniquedates.pickle")[0]
allvocs = import_pickle("E://Summer research/pythonProject/data/allvocs.pickle")[0]

'''Create a NETCDF4 file'''
def create_netcdf(filepath):
    #create a NETCDF4 file in the data directory of this project
    nc_file = netCDF4.Dataset(filepath,mode='w',format='NETCDF4')
    #assign the 3 dimensions with names & sizes
    voc_dim = nc_file.createDimension('VOC',size=176)
    time_dim = nc_file.createDimension('Time',size=11322) #Time is an unlimited dimension
    station_dim = nc_file.createDimension('Station',size=None)
    nc_file.title = 'VOC Concentrations across Canada (1989-2019)'
    nc_file.subtitle = 'Raw Data'

    '''Define variables for each axis'''
    voc_var = nc_file.createVariable('VOC',np.str_,('VOC',))
    voc_var.long_name = 'Volatile Organic Compounds'

    time_var = nc_file.createVariable('Time',np.str_,('Time',))
    time_var.units = 'Date in format YYYY-MM-DD'

    station_var = nc_file.createVariable('Station', np.str_, ('Station'))
    #Create a 3D variable for concentration
    conc_var = nc_file.createVariable('Conc.',np.float64,('Time','VOC','Station'))
    conc_var.long_name = 'Concentration'
    conc_var.units = '(ug/m3)'
    return nc_file

def get_indices():
    dates = np.arange(np.datetime64('1989-01-01'),np.datetime64('2020-01-01'))
    numbers = np.arange(0,len(dates))
    dates_df=pd.DataFrame(dates)
    num_df = pd.DataFrame(numbers)
    vocs_df = pd.DataFrame(allvocs)
    indices = pd.concat([num_df,dates_df,vocs_df],axis=1)
    indices.columns=['Index','Date','VOC']
    return indices

def average_duplicate_dates(df):
    '''Removes duplicate dates and instead averages the concentrations for that date into one date'''
    dates = df['Compounds']
    boolseries = dates.duplicated(keep='first')
    indices = boolseries[boolseries==True].index.to_numpy()
    if indices.size!=0:
        groups = np.split(indices,np.where(np.diff(indices)!=1)[0]+1)
        for array_group in groups:
            first_duplicate_index = array_group[0]-1
            last_duplicate_index = array_group[-1]
            rows = df.iloc[first_duplicate_index:last_duplicate_index+1,1:]
            sum_rows = rows.sum(axis=0,skipna=True, min_count=0)
            bool_nan = rows.isna().all()
            mean = sum_rows/(len(array_group)+1)
            mean[bool_nan==True]=np.nan
            df.iloc[first_duplicate_index:last_duplicate_index+1,1:]=mean
            df.drop_duplicates(subset=['Compounds'],keep='first',inplace=True,ignore_index=True)
    return df

ncfile = create_netcdf("E://Summer research/pythonProject/data/my_data2.nc")

indices = get_indices()

df = alldata['S100110']

# #add data 1
stations = np.array(list(alldata.keys()))
ncfile['Station'][:]=stations

# #add data 2
ncfile['VOC'][:] = allvocs

ncfile['Time'][:]=indices['Date'].to_string()

df['Compounds'] = df['Compounds'].to_string()

ncfile['Conc.'][:,:,0]=df.iloc[:,:]


#
# voc_indices = {}
# i=0
# for voc in allvocs:
#     voc_indices[voc]=i
#     i+=1
#
# station_indices={}
# i=0
# for station in stations:
#     station_indices[station]=i
#     i+=1
#
# df = alldata['S100110']
# #time_var[0:len(df)]=date2num(df['Compounds'],time_var.units,'gregorian')
# time_var[0:len(df)-1]=df['Compounds'].astype(str)
# conc_var[:,station_indices['S100110'],voc_indices['Ethane']]=df['Ethane']
#print(nc_file)

# time_indices = indices['Index'][indices['Date']==df['Compounds']]
# indices_dates = np.where(np.searchsorted(df2,dates,'left')!=np.searchsorted(df2,dates,'right'))[0]

'''"E://Summer research/pythonProject/data/my_data.nc"'''
