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
from scatter import find_regular_medians

