from DataCleaning import import_pickle, export_pickle
import pandas as pd

class Station:
    def __init__(self,ID,dataframe):
        '''
        Initializes the Station class.
        input: ID (str) --> ID of station. Eg. "S010102"
               dataframe (dict) --> A dictionary containing collapsed Dataframes for each station. Must be collapsed
               and have repeated columns merged using merge_repeated().

               Eg. {station1:df1, station2:df2, ..., station(i):df(i)},
                where each df is of the format:
                eg. station1: (time goes from years 1989-2019/start-end)
                |   Compounds | VOC1  | VOC2 | ... |
                ----------------------------------------
                |   1/1/1989  | 1.346 | 2.345| ... |
                |    ...      |  ...  |  ... | ... |
                |    ...      |  ...  |  ... | ... |
                |  31/12/2019 | 2.669 | 5.876| ... |

        output: Station (obj) with attributes ID, data, metadata. ID (str), data (df), metadata(df)
        '''
        self.ID = ID
        self.data = dataframe[ID] #get the dataframe for the particular station ID
        #get the metadata from an external dataframe source
        metadata = import_pickle('E://Summer research/pythonProject/data/metadata.pickle')[0]
        #set the urbanization attribute to whatever is assigned to this station ID in the metadata df
        self.urbanization = (metadata['Urbanization'][metadata['NAPS ID']==ID]).iloc[0]
    def get_voc(self,voc,start,end):
        '''
        Gives a DataFrame with dates & measurements for a specific VOC in a given year range.
        input: voc (str) --> name of voc. Eg. 'Ethane'
               start (str) --> start year. Eg. '1989'
               end (str) --> end year. Eg. '2000'

        output: arr (df) --> Eg. station1 = Station('S010102',dataframe)
        station1.get_voc(voc = 'Ethane', start='1989', end='2013')
        |   Compounds | Ethane|
        -----------------------
        |   1/1/1989  | 1.346 |
        |    ...      |  ...  |
        |    ...      |  ...  |
        |  01/01/2013 | 2.669 |
        '''
        df = self.data
        start_yr = pd.to_datetime('{}-01-01'.format(start),infer_datetime_format=True) #get the start year as a Datetime object
        end_yr = pd.to_datetime('{}-01-01'.format(end),infer_datetime_format=True) #does not include the end year
        arr = df[['Compounds',voc]][(df['Compounds']>=start_yr) & (df['Compounds']<end_yr)]
        return arr
