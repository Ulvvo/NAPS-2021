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

def find_regular_medians(df,interval,voc):
    '''Takes in a DataFrame for a station and finds the medians for arrays of data for regular intervals of time.
    Eg. Every 5 years or every 10 years.
    '''
    medians={}
    if interval==5:
        ranges = [(1989, 1994), (1994, 1999), (1999, 2004),(2004,2009),(2009,2014),(2014,2020)]
    else:
        ranges = [(1989, 1999), (1999, 2009), (2009, 2020)]
    for start,end in ranges:
        start_yr = pd.to_datetime('{}-01-01'.format(start), infer_datetime_format=True)
        end_yr = pd.to_datetime('{}-01-01'.format(end), infer_datetime_format=True)
        try:
            arr = df[voc][(df['Compounds'] >= start_yr) & (df['Compounds'] < end_yr)]
            med = np.nanmedian(arr)
            medians[start]=med
        except KeyError:
            pass
    return medians

def create_scatter_data(data,voc,interval):
    '''Returns a dataframe containing all the data pieces for a scatter/regression graph.'''
    # md_selected = md[['NAPS ID', 'Province', 'Population density', 'Urbanization', 'Site Type', 'Land Use']][
    #     md['Site Type'] != 'PS']
    # emptarr = np.empty((len(md_selected), 1))
    # emptarr[:] = np.nan
    # md_selected['Year'] = emptarr
    # md_selected['{}'.format(voc)] = emptarr
    scatter_df = pd.DataFrame(columns=['NAPS ID', 'Province', 'Population density (1991 census)',
                                       'Population density (1996 census)','Population density (2001 census)',
                                       'Population density (2006 census)', 'Population density (2011 census)',
                                       'Population density (2016 census)',
                                       'Urbanization', 'Site Type',
                                       'Land Use','Year',voc])
    rows_collection=[scatter_df]
    for station in data:
        df = data[station]
        medians=find_regular_medians(df,interval,voc)
        try:
            province = md['Province'][md['NAPS ID']==station].iloc[0]
            urbanization = md['Urbanization'][md['NAPS ID']==station].iloc[0]
            site_type = md['Site Type'][md['NAPS ID']==station].iloc[0]
            land_use = md['Land Use'][md['NAPS ID']==station].iloc[0]
            if len(medians)!=0:
                for year in medians:
                    pop_density = md['Population density ({} census)'.format(year+2)][md['NAPS ID'] == station].iloc[0]
                    row_to_add = pd.DataFrame({'NAPS ID':station,'Province':province,
                                               'Population density ({} census)'.format(year+2):pop_density,
                                               'Urbanization':urbanization,'Site Type':site_type,'Land Use':land_use,
                                               'Year':year,voc:medians[year]},index=[0])
                    rows_collection.append(row_to_add)
        except IndexError:
            pass
    scatter_df=pd.concat(rows_collection,axis=0,ignore_index=True)
    return scatter_df

def create_scatter(data,voc,interval,start=None):
    '''Creates a scatterplot with population density vs. median concentration of a given VOC for a certain timeframe.
    data --> colldatav4
    '''
    # md_selected = md[['NAPS ID','Province','Population density','Urbanization','Site Type','Land Use']][md['Site Type']!='PS']
    # emptarr = np.empty((len(md_selected), 1))
    # emptarr[:] = np.nan
    # md_selected['{}'.format(start)] = emptarr
    # for station in data:
    #     df = data[station]
    #     medians = find_regular_medians(df,interval,voc)
    #     if len(medians)!=0:
    #         md_selected['{}'.format(start)][md_selected['NAPS ID'] == station] = medians[start]
    scatter_data = create_scatter_data(data, voc, interval)
    fig,ax=plt.subplots()
    font = {'family': 'serif',
            # 'color': 'black',
            'weight': 'normal',
            'size': 10,
            }
    #'grid.color':'grey',
    sns.set_style({'axes.facecolor':'#e8e8ea','axes.grid':True,'font.family':'Serif'})
    sns.scatterplot(ax=ax,x='Population density', y='{}'.format(voc), data=scatter_data, hue='Year',alpha=0.5,s=80)
    ax.set_xlabel('Population density (people/sq km)', fontdict=font)
    ax.set_ylabel('Median {} Concentration (ug/m3)'.format(voc),fontdict=font)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.set_xlim(0,6000)
    ax.set_ylim(bottom=0)
    ax.grid(b=True,which='major',axis='x',color='#d8d8df')
    if start==None:
        plt.suptitle(
            'Median concentration of {} vs. Population density \n (All years)'.format(voc))
    else:
        plt.suptitle('Median concentration of {} vs. Population density \n Years {}-{}'.format(voc,start,start+interval))
    ax.legend(loc='upper right',title='Period starting',fancybox=False)
    plt.tight_layout()
    plt.show()
    return scatter_data

def create_regression(data,voc,interval,start=None,path=None):
    '''Creates a scatterplot with population density vs. median concentration of a given VOC for a certain timeframe.
    data --> colldatav4
    '''
    scatter_data = create_scatter_data(data,voc,interval)
    fig,ax=plt.subplots()
    font = {'family': 'serif',
            # 'color': 'black',
            'weight': 'normal',
            'size': 10,
            }
    sns.set_style({'axes.facecolor':'#eaeaee','axes.grid':True,'grid.color':'grey','font.family':'Serif'})
    if start==None:
        sns.regplot(ax=ax, x='Population density', y='{}'.format(voc), data=scatter_data,x_ci=95,ci=95,scatter=True,fit_reg=True,scatter_kws={'color':'purple','alpha':0.5,'s':80})
    else:
        sns.regplot(ax=ax, x='Population density', y='{}'.format(voc), data=scatter_data[scatter_data['Year']==start], x_ci=95, ci=95, scatter=True,
                    fit_reg=True, scatter_kws={'color': 'purple', 'alpha': 0.5, 's': 80})
    ax.set_xlabel('Population density (people/sq km)', fontdict=font)
    ax.set_ylabel('Median {} Concentration (ug/m3)'.format(voc), fontdict=font)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
    ax.set_xlim(0,6000)
    ax.set_ylim(bottom=0)
    ax.grid(b=True,which='major',axis='x',color='#eaeaee')
    if start==None:
        plt.suptitle(
            'Median concentration of {} vs. Population density \n (All years)'.format(voc))
    else:
        plt.suptitle('Median concentration of {} vs. Population density \n years {}-{}'.format(voc,start,start+interval))
    plt.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
    return scatter_data

def linear_regression(data,voc,interval,year):
    '''Creates a linear regression model of population density related to concentration of species.'''
    df = create_scatter_data(data,voc,interval)
    selected_data = df[df['Year'] == year]
    selected_data=selected_data[['Population density ({} census)'.format(year+2), voc]].dropna(axis=0, how='any')
    #get the pop density in thousands
    X = selected_data['Population density ({} census)'.format(year+2)]
    Y = selected_data[voc]
    X = sm.add_constant(X)
    if len(Y)>=5:
        model = sm.OLS(Y,X).fit()
        return model
    else:
        return None

'''Get the lambda transformation value for the population densities'''
popdensity_lambda = {1989: 0.22626743969386043, 1994: 0.20650893856262537, 1999: 0.20522302371282758,
                     2004: 0.21932505220469775, 2009: 0.2352741294835062, 2014: 0.21985046474681946}
popdensity_mean = 0.21874150806738948

def linear_regression_transformed(data,voc,interval,year):
    '''Creates a linear regression model of population density related to concentration of species.'''
    df = create_scatter_data(data,voc,interval)
    xy = df[['Population density ({} census)'.format(year + 2), voc]][df['Year'] == year].dropna(axis=0, how='any')
    xy_nonzero = xy[(xy[voc] != 0) & (xy['Population density ({} census)'.format(year + 2)] != 0)]
    if len(xy_nonzero) > 5:
        x, l1 = scipy.stats.boxcox(xy_nonzero['Population density ({} census)'.format(year + 2)])
        y, l2 = scipy.stats.boxcox(xy_nonzero[voc])
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return model,l1,l2
    else:
        return None,None,None

def create_regression2(data,voc,interval,path=None):
    '''Creates a scatter plot with population density vs. median concentration of a given VOC for a certain timeframe.
    data --> colldatav4
    '''
    scatter_data = create_scatter_data(data,voc,interval)
    fig,ax=plt.subplots(nrows=2,ncols=3,figsize=(80,40))
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 30,
            }
    font_title= {'family': 'serif',
            'weight': 'semibold',
            'size': 40,
            }
    sns.set_style({'axes.facecolor':'white','axes.grid':True,'grid.color':'white','font.family':'Serif'})
    years = [(1989,1994),(1994,1999),(1999,2004),(2004,2009),(2009,2014),(2014,2020)]
    census_years = [1991,1996,2001,2006,2011,2016]
    row_col_nums = {0:(0,0),1:(0,1),2:(0,2),3:(1,0),4:(1,1),5:(1,2)}
    for i in range(6):
        selected_data = scatter_data[scatter_data['Year']==years[i][0]]
        selected_data_cleaned = selected_data[['Population density ({} census)'.format(census_years[i]),voc]].dropna(axis=0,how='any')
        row = row_col_nums[i][0]
        col = row_col_nums[i][1]
        try:
            sns.scatterplot(ax=ax[row,col],x='Population density ({} census)'.format(census_years[i]), y='{}'.format(voc), data=selected_data, hue='Province',alpha=0.5,s=200)
            x,y = (selected_data_cleaned['Population density ({} census)'.format(census_years[i])],selected_data_cleaned[voc])
            X=sm.add_constant(x)
            model = sm.OLS(y,X).fit()
            ci = model.conf_int()
            params = model.params
            (r2,b0,b0_ci_l,b0_ci_u,b1,b1_ci_l,b1_ci_u)=(model.rsquared,params[0],ci.iloc[0,0],ci.iloc[0,1],params[1],ci.iloc[1,0],ci.iloc[1,1])
            yhat = b0+b1*x
            yhat_lower = b0_ci_l+b1_ci_l*x
            yhat_upper = b0_ci_u+b1_ci_u*x
            sns.lineplot(ax=ax[row,col],x=x,y=yhat,palette='Set2',color='Blue',label='{}+{:.2e}x, R2 ={}'.format(round(b0,3),b1,round(r2,3)))
            sns.lineplot(ax=ax[row,col], x=x, y=yhat_lower, palette='Set2', color='Green',linestyle='--')
            sns.lineplot(ax=ax[row,col], x=x, y=yhat_upper, palette='Set2', color='Green',linestyle='--')
            # ax[i].fill_between(np.arange(0,6000),yhat_upper, yhat_lower, facecolor='blue', alpha=0.15)
            ax[row,col].set_xlabel('Population density (people/sq km)', fontdict=font)
            ax[row,col].set_ylabel('Median {} Concentration (ug/m3)'.format(voc), fontdict=font)
            ax[row,col].set_title('Years {}-{}'.format(years[i][0],years[i][1]),fontproperties=font)
            ax[row,col].yaxis.set_major_locator(mpl.ticker.MaxNLocator(20))
            ax[row,col].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax[row,col].xaxis.set_major_locator(mpl.ticker.MaxNLocator(10))
            ax[row,col].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
            ax[row,col].set_xlim(0,6000)
            ax[row,col].set_ylim(bottom=0)
            ax[row,col].grid(b=True,which='major',axis='x',color='#eaeaee')
        except ValueError:
            pass
        except IndexError:
            pass
    plt.suptitle('Median concentration of {} vs. Population density over different 5-year periods'.format(voc),fontproperties=font_title)
    plt.tight_layout()
    plt.legend(loc='lower right',fontsize='large')
    if path==None:
        plt.show()
    else:
        plt.savefig(path)
    return scatter_data

def residual_displot(model,path=None):
    '''Returns a residual histogram plot'''
    residuals = model.resid
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20,
            }
    sns.distplot(a=residuals,kde=True)
    plt.xlabel('Residuals (model)',fontdict=font)
    plt.title('Frequency distribution of model error residuals',fontdict=font)
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

def residual_qq(model,path=None):
    '''Returns a residual qq plot'''
    residuals = model.resid
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20,
            }
    sm.qqplot(residuals)
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

def residual_scatter(model,voc,start,end,path=None):
    '''Returns a residual qq plot'''
    residuals = model.resid
    fitted = model.fittedvalues
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20,
            }
    x = np.linspace(np.min(fitted),np.max(fitted),100)
    y=np.zeros(100)
    sns.scatterplot(x=fitted,y=residuals)
    sns.lineplot(x=x,y=y)
    plt.xlabel('Fitted concentration (ug/m3)')
    plt.ylabel('Residual error')
    plt.title('Residual vs. fitted values for {}, {}-{}'.format(voc,start,end))
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

def residual_both(model,voc,start,end,path=None):
    '''Returns both a residual histogram plot & qq plot'''
    fig,ax=plt.subplots(nrows=1,ncols=2)
    residuals = model.resid
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 10,
            }
    font_title = {'family': 'serif',
            'weight': 'bold',
            'size': 11,
            }
    sns.histplot(ax=ax[0],x=residuals, stat='frequency',kde=True,edgecolor='None')
    ax[0].set_xlabel('Residuals (model)',fontdict=font)
    ax[0].set_ylabel('Frequency', fontdict=font)
    ax[0].set_title('Distribution of residuals',fontdict=font)
    sm.qqplot(ax=ax[1],data=residuals,fit=True,line='r',marker='o',markerfacecolor="None",
         markeredgecolor='blue')
    # x = np.array([-2,-1,0,1,2])
    # y = np.array([-2,-1,0,1,2])
    # sns.lineplot(ax=ax[1],x=x,y=y)
    ax[1].set_xlabel('Normal theoretical quantiles', fontdict=font)
    ax[1].set_ylabel('Sample quantiles', fontdict=font)
    ax[1].set_title('Q-Q plot of residuals',fontdict=font)
    plt.suptitle('Analysis of residuals for {} ({}-{})'.format(voc,start,end),fontproperties=font_title,y=0.97)
    plt.tight_layout()
    if path==None:
        plt.show()
    else:
        plt.savefig(path)

def benzene_ER_data(data,voc,interval):
    '''Returns a DataFrame with the Benzene enhancement ratios of each station over all the years.'''
    ER_df = pd.DataFrame(columns=['NAPS ID', 'Province', 'Population density (1991 census)',
                                       'Population density (1996 census)', 'Population density (2001 census)',
                                       'Population density (2006 census)', 'Population density (2011 census)',
                                       'Population density (2016 census)',
                                       'Urbanization', 'Site Type',
                                       'Land Use', 'Year', '{} ER'.format(voc)])
    rows_collection = [ER_df]
    for station in data:
        df = data[station]
        try:
            RB_station = md['Regional background'][md['NAPS ID']==station].iloc[0]
            df_rb = data[RB_station]
            voc_station_medians = find_regular_medians(df, interval, voc)
            benzene_station_medians=find_regular_medians(df, interval, 'Benzene')
            voc_RB_medians = find_regular_medians(df_rb, interval, voc)
            benzene_RB_medians = find_regular_medians(df_rb, interval, 'Benzene')
            all_meds = [benzene_station_medians,benzene_RB_medians,voc_station_medians,voc_RB_medians]
            non_empty = [i for i in all_meds if len(i)!=0]
            province = md['Province'][md['NAPS ID'] == station].iloc[0]
            urbanization = md['Urbanization'][md['NAPS ID'] == station].iloc[0]
            site_type = md['Site Type'][md['NAPS ID'] == station].iloc[0]
            land_use = md['Land Use'][md['NAPS ID'] == station].iloc[0]
            if len(non_empty)==4:
                for year in voc_station_medians:
                    pop_density = md['Population density ({} census)'.format(year + 2)][md['NAPS ID'] == station].iloc[0]
                    benzene_difference = benzene_station_medians[year]-benzene_RB_medians[year]
                    voc_difference = voc_station_medians[year]-voc_RB_medians[year]
                    ER = voc_difference/benzene_difference
                    row_to_add = pd.DataFrame({'NAPS ID': station, 'Province': province,
                                               'Population density ({} census)'.format(year + 2): pop_density,
                                               'Urbanization': urbanization, 'Site Type': site_type,
                                               'Land Use': land_use,
                                               'Year': year, '{} ER'.format(voc): ER}, index=[0])
                    rows_collection.append(row_to_add)
        except KeyError:
            pass
        except IndexError:
            pass
    ER_df = pd.concat(rows_collection, axis=0, ignore_index=True)
    return ER_df

def benzene_ER_graph(ER_data,voc,year,path=None):
    '''Takes in a DataFrame of Benzene ER data and produces a regression graph for it.'''
    selected_data = ER_data[ER_data['Year'] == year]
    selected_data_cleaned = selected_data[['Population density ({} census)'.format(year+2), '{} ER'.format(voc)]].dropna(axis=0, how='any')
    sns.scatterplot(x='Population density ({} census)'.format(year+2), y='{} ER'.format(voc),data=selected_data)
    x, y = (selected_data_cleaned['Population density ({} census)'.format(year+2)], selected_data_cleaned['{} ER'.format(voc)])
    if len(x)>=5:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        ci = model.conf_int()
        params = model.params
        (r2, b0, b0_ci_l, b0_ci_u, b1, b1_ci_l, b1_ci_u) = (
        model.rsquared, params[0], ci.iloc[0, 0], ci.iloc[0, 1], params[1], ci.iloc[1, 0], ci.iloc[1, 1])
        yhat = b0 + b1 * x
        yhat_lower = b0_ci_l + b1_ci_l * x
        yhat_upper = b0_ci_u + b1_ci_u * x
        sns.lineplot(x=x, y=yhat, palette='Set2', color='Blue',
                     label='{}+{:.2e}x, R2 ={}'.format(round(b0, 3), b1, round(r2, 3)))
        sns.lineplot(x=x, y=yhat_lower, palette='Set2', color='Green', linestyle='--')
        sns.lineplot(x=x, y=yhat_upper, palette='Set2', color='Green', linestyle='--')
        plt.xlabel('Population density (people/sq km)')
        plt.ylabel('Median {} ER'.format(voc))
        plt.title('{} ER: years {}-{}'.format(voc,year,year+5))
        plt.tight_layout()
        if path==None:
            plt.show()
        else:
            plt.savefig(path)
            plt.clf()

def ER_model(ER_df,voc,year):
    selected_data = ER_df[ER_df['Year'] == year]
    selected_data = selected_data[['Population density ({} census)'.format(year + 2), '{} ER'.format(voc)]].dropna(axis=0, how='any')
    # get the pop density in thousands
    X = selected_data['Population density ({} census)'.format(year + 2)]
    Y = selected_data['{} ER'.format(voc)]
    X = sm.add_constant(X)
    if len(Y) >= 5:
        model = sm.OLS(Y, X).fit()
        return model
    else:
        return None

def get_regression_models():
    rows_all = []
    missed_rows=[]
    for voc in VCPs:
        for year,end in years:
                model=linear_regression(alldata,voc,5,year)
                if model!=None:
                    N = round(model.nobs)
                    R2 = model.rsquared
                    b1=model.params.iloc[1]
                    b1_pval = model.pvalues.iloc[1]
                    b1_lower=model.conf_int().iloc[1, 0]
                    b1_upper=model.conf_int().iloc[1, 1]
                    b0=model.params.iloc[0]
                    b0_pval = model.pvalues.iloc[0]
                    b0_lower = model.conf_int().iloc[0, 0]
                    b0_upper = model.conf_int().iloc[0, 1]
                    mse = model.mse_resid
                    rmse = np.sqrt(mse)
                    row_to_add = pd.DataFrame({'VOC':voc,'Year':year,'N':N,'R^2':R2,'b1: Pop. density':b1,'b1: P > |t|':b1_pval,
                                               'b1 lower CI':b1_lower,'b1 upper CI':b1_upper,'b0: constant':b0,'b0: P > |t|':b0_pval,
                                               'b0 lower CI':b0_lower,'b0 upper CI':b0_upper,'RMSE':rmse},index=[0])
                    rows_all.append(row_to_add)
    df=pd.concat(rows_all,axis=0,ignore_index=True)
    return df

def get_regression_models_transformed():
    rows_all = []
    missed_rows=[]
    for voc in VCPs:
        for year,end in years:
                model,l1,l2=linear_regression_transformed(alldata,voc,5,year)
                if model!=None:
                    N = round(model.nobs)
                    R2 = model.rsquared
                    b1=model.params[1]
                    b1_pval = model.pvalues[1]
                    b1_lower=model.conf_int()[1, 0]
                    b1_upper=model.conf_int()[1, 1]
                    b0=model.params[0]
                    b0_pval = model.pvalues[0]
                    b0_lower = model.conf_int()[0, 0]
                    b0_upper = model.conf_int()[0, 1]
                    mse = model.mse_resid
                    rmse = np.sqrt(mse)
                    row_to_add = pd.DataFrame({'VOC':voc,'Year':year, 'lambda (x)':l1,'lambda (y)':l2,'N':N,'R^2':R2,'b1: Pop. density':b1,'b1: P > |t|':b1_pval,
                                               'b1 lower CI':b1_lower,'b1 upper CI':b1_upper,'b0: constant':b0,'b0: P > |t|':b0_pval,
                                               'b0 lower CI':b0_lower,'b0 upper CI':b0_upper,'RMSE':rmse},index=[0])
                    rows_all.append(row_to_add)
    df=pd.concat(rows_all,axis=0,ignore_index=True)
    return df

def spearmans(voclist):
    spearman_df = pd.DataFrame(columns=['VOC','Year','Spearman Coefficient','P-val'])
    rows_to_add = [spearman_df]
    for voc in voclist:
        er_df = benzene_ER_data(alldata,voc,5)
        for year,end in years:
            new_df = er_df[['Population density ({} census)'.format(year+2),'{} ER'.format(voc)]][er_df['Year']==year].dropna(axis=0,how='any')
            x = new_df['Population density ({} census)'.format(year + 2)]
            y = new_df['{} ER'.format(voc)]
            sp = scipy.stats.spearmanr(x, y)
            df_to_add = pd.DataFrame({'VOC':voc,'Year':year,'Spearman Coefficient':sp.correlation,'P-val':sp.pvalue},index=[0])
            rows_to_add.append(df_to_add)
    new_df = pd.concat(rows_to_add, axis=0, ignore_index=True)
    return new_df

def pearsons(voclist):
    pearson_df = pd.DataFrame(columns=['VOC','Year','N','Pearson Coefficient','P-val'])
    rows_to_add = [pearson_df]
    for voc in voclist:
        scatter_data = create_scatter_data(alldata,voc,5)
        for year,end in years:
            new_df = scatter_data[['Population density ({} census)'.format(year+2),voc]][(scatter_data['Year']==year)&(scatter_data['Population density ({} census)'.format(year+2)]!=0)&(scatter_data[voc]!=0)].dropna(axis=0,how='any')
            x = np.log10(new_df['Population density ({} census)'.format(year + 2)])
            y = np.log10(new_df[voc])
            try:
                if len(x)>5:
                    psr = scipy.stats.pearsonr(x, y)
                    df_to_add = pd.DataFrame({'VOC':voc,'Year':year,'N':len(x),'Pearson Coefficient':psr[0],'P-val':psr[1]},index=[0])
                    rows_to_add.append(df_to_add)
            except ValueError:
                raise Exception('Some infinity/NaN value for {},{}'.format(voc,year))
    new_df = pd.concat(rows_to_add, axis=0, ignore_index=True)
    return new_df

def transformed_scatter(alldata,voc,interval,year,path=None):
    scatter_df = create_scatter_data(alldata,voc,5)
    xy = scatter_df[['Population density ({} census)'.format(year + 2), voc]][scatter_df['Year'] == year].dropna(
        axis=0, how='any')
    xy_nonzero = xy[(xy[voc] != 0) & (xy['Population density ({} census)'.format(year + 2)] != 0)]
    if len(xy_nonzero)>5:
        x,l1 = scipy.stats.boxcox(xy_nonzero['Population density ({} census)'.format(year + 2)])
        y,l2 = scipy.stats.boxcox(xy_nonzero[voc])
        sns.scatterplot(x=x,y=y)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        ci = model.conf_int()
        params = model.params
        (r2, b0, b0_ci_l, b0_ci_u, b1, b1_ci_l, b1_ci_u) = (
            model.rsquared, params[0], ci[0, 0], ci[0, 1], params[1], ci[1, 0], ci[1, 1])
        yhat = b0 + b1 * x
        yhat_lower = b0_ci_l + b1_ci_l * x
        yhat_upper = b0_ci_u + b1_ci_u * x
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 10,
                }
        font_title = {'family': 'serif',
                      'weight': 'bold',
                      'size': 11,
                      }
        sns.lineplot(x=x, y=yhat, palette='Set2', color='Blue')
        label = 'y={}+{:.2e}x \n R2 ={}'.format(round(b0, 3), b1, round(r2, 3))
        sns.lineplot(x=x, y=yhat_lower, palette='Set2', color='Green', linestyle='--')
        sns.lineplot(x=x, y=yhat_upper, palette='Set2', color='Green', linestyle='--')
        plt.title('Population density vs. Median Concentration of {} \n {}-{}'.format(voc,year,year+5),fontdict=font_title)
        plt.xlabel('Population density (transformed)',fontdict=font)
        plt.ylabel('Median Concentration (transformed)'.format(voc),fontdict=font)
        plt.legend(loc='upper left', title='Legend',handles=[mpl.lines.Line2D([],[],linestyle='--',color='green',label='95% CI'),mpl.lines.Line2D([],[],linestyle='-',color='blue',label='{}'.format(label))],prop=font)
        if path == None:
            plt.show()
        else:
            plt.savefig(path)
            plt.clf()
        return (l1,l2)
    else:
        return (None,None)

def ER_time_graphs(er_df,voc,path=None):
    # i=0
    # rows = int(len(er_df[er_df['Urbanization']==urbanization])/6)
    # fig,ax = plt.subplots(nrows=rows,ncols=1,figsize=(40,10))
    fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))
    coords = [(0,0),(0,1),(1,0),(1,1)]
    colours = {'ON':'green','QC':'red','AB':'pink','BC':'blue','SK':'yellow','NB':'gray','NS':'orange','MB':'black'}
    for i,j in coords:
        pos = coords.index((i,j))
        urbanization = urbanizations[pos]
        for station in alldata:
            try:
                df_graph = er_df[['Year','{} ER'.format(voc),'Province']][(er_df['Urbanization']==urbanization)&(er_df['NAPS ID']==station)].dropna(axis=0,how='any')
                if len(df_graph)>=3:
                    province = df_graph['Province'].iloc[0]
                    # sns.scatterplot(ax=ax[i,j],x='Year', y='{} ER'.format(voc), data=df_graph,palette='Set2')
                    sns.lineplot(ax=ax[i,j],x='Year',y='{} ER'.format(voc),data=df_graph,marker='o',palette='Set2',color='{}'.format(colours[province]),linestyle='--')
                    handles = [mpl.lines.Line2D([],[],color='green',label='ON'),
                               mpl.lines.Line2D([],[],color='red',label='QC'),
                              mpl.lines.Line2D([],[],color='pink',label='AB'),
                                mpl.lines.Line2D([],[],color='blue',label='BC'),
                               mpl.lines.Line2D([],[],color='yellow',label='SK'),
                                mpl.lines.Line2D([],[],color='gray',label='NB'),
                                mpl.lines.Line2D([],[],color='orange',label='NS'),
                                mpl.lines.Line2D([],[],color='black',label='MB')]
            except ValueError:
                pass
        ax[i,j].legend(loc='upper left', title='Province', handles=handles, fancybox=False, ncol=2)
        ax[i,j].set_title('{}'.format(urbanization))
    plt.suptitle('{} ER over the years'.format(voc),y=0.98)
    plt.legend()
    plt.tight_layout()
    if path == None:
        plt.show()
    else:
        plt.savefig(path)
        plt.clf()





urbanizations = ['LU','MU','SU','NU']
# scatter_data=create_scatter(alldata,'1,4-Dichlorobenzene',5)
# reg_data = create_regression(alldata,'1,4-Dichlorobenzene',5)
# ms = linear_regression(alldata,'Ethylbenzene',5)

# md_sel2=create_scatter(alldata,'Acetone',5,2009)

alldata = import_pickle("E://Summer research/pythonProject/data/colldatav4.pickle")[0]
md = pd.read_excel("E:\VOC Project\Random\StationsMetadata.xlsx")
# ms = linear_regression(alldata,'Ethane',5,1989)
#
VCPs = ['d-Limonene','Camphene','Acetone','Propane','Butane','Isopentane','Cyclopentane','Pentane','Cyclohexane',
        'Heptane','Methylcyclohexane','Toluene','Octane','Ethylbenzene','Styrene','Decane','Undecane','Dodecane',
        'Bromomethane','Dichloromethane','Tetrachloroethylene','1,4-Dichlorobenzene','1,3-Dichlorobenzene',
        '1,2-Dichlorobenzene','MIBK']

# for voc in VCPs:
#     er_df = benzene_ER_data(alldata,voc,5)
#     ER_time_graphs(er_df,voc,path="E://Summer research/pythonProject/figures/scatter/ER time/{}.png".format(voc))

# vocs=['Toluene','Octane','Ethylbenzene','Styrene','Undecane','Dodecane',
#         'Bromomethane','Dichloromethane','Tetrachloroethylene','1,4-Dichlorobenzene']
# vocs_set = set(vocs)
# vcps_set=set(VCPs)
# leftover = vcps_set.difference(vocs_set)
#scatter_data = create_regression2(alldata,'d-Limonene',5,"E://Summer research/pythonProject/figures/scatter/Regression 2/trial.png")

years = [(1989,1994),(1994,1999),(1999,2004),(2004,2009),(2009,2014),(2014,2020)]
voc='Dichloromethane'
interval=5
# rm=get_regression_models_transformed()
df = create_scatter_data(alldata, voc, interval)
# lambdas = {}
# for voc in VCPs[19:]:
#     for year,end in years:
#         l1,l2=transformed_scatter(alldata,voc,5,year,"E://Summer research/pythonProject/figures/scatter/Transformed/{},{}.png".format(voc,year))
#         lambdas[voc]=(year,l1,l2)

# sd = create_scatter_data(alldata,'Ethane',5)
# voc = 'Dichloromethane'
# transformed_scatter(alldata,voc,5,2009)
# voc1=['Benzene']
# year=1989
#
# er_df = benzene_ER_data(alldata,voc,5)
# ER_time_graphs(er_df,voc)
# df = ER_model(er_df,voc,year)
# model = linear_regression(alldata,voc,5,year)
# residual_both(model,voc,year,year+5)
# residual_scatter(model,voc,start=year,end=year+5)
# er_df = benzene_ER_data(alldata,voc,5)
# benzene_ER_graph(er_df,voc,1999)
# model=linear_regression(alldata,voc,5,year)
# residual_both(model,voc,year,year+5,'E://Summer research/pythonProject/figures/scatter/Residuals/{},{}.png'.format(voc,year))
# df = pd.DataFrame(columns=['VOC','Year','N','R^2','b1: Pop. density','b1: P > |t|','b1 lower CI','b1 upper CI',
#                            'b0: constant','b0: P > |t|','b0 lower CI','b0 upper CI','RMSE'])

# for voc in VCPs:
#     er_df = benzene_ER_data(alldata, voc, 5)
#     for year,end in years:
#         ER_model(er_df,voc,year)
        #benzene_ER_graph(er_df,voc,year,path="E://Summer research/pythonProject/figures/scatter/ER/{},{}.png".format(voc,year))

# for voc in vocs:
#     for year,end in years:
#         model = linear_regression(alldata, voc, 5, year)
#         residual_scatter(model,voc,start=year,end=end,path='E://Summer research/pythonProject/figures/scatter/Residuals2/{},{}.png'.format(voc,year))
    #create_regression2(data=alldata, voc=voc, interval=5,path="E://Summer research/pythonProject/figures/scatter/Regression - revised/{}.png".format(voc))

# df = get_regression_models()
# er_df = benzene_ER_data(alldata, voc, 5)
# df=spearmans(VCPs)
# df2 = pearsons(VCPs)

#df_graph = er_df[['Year','{} ER'.format(voc),'Province']][(er_df['Urbanization']==urbanization)&(er_df['Province']==province)&(er_df['NAPS ID']==station)].dropna(axis=0,how='any')