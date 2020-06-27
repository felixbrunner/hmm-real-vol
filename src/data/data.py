import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import requests
import io
import zipfile


def download_factor_data(freq='D'):

    '''
    Downloads factor data from Kenneth French's website and returns dataframe.
    freq can be either 'D' (daily) or 'M' (monthly).
    '''

    if freq is 'D':
        # Download Carhartt 4 Factors
        factors_daily = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start='1/1/1900')[0]
        mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start='1/1/1900')[0]
        factors_daily = factors_daily.join(mom)
        factors_daily = factors_daily[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_daily.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        return factors_daily

    elif freq is 'M':
        # Download Carhartt 4 Factors
        factors_monthly = web.DataReader("F-F_Research_Data_Factors", "famafrench", start='1/1/1900')[0]
      #  mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start='1/1/1900')[0] #There seems to be a problem with the data file, fix if mom is needed
      #  factors_monthly = factors_monthly.join(mom)
      #  factors_monthly = factors_monthly[['Mkt-RF','SMB','HML','Mom   ','RF']]
        factors_monthly.index = factors_monthly.index.to_timestamp()
      #  factors_monthly.columns = ['Mkt-RF','SMB','HML','Mom','RF']
        factors_monthly.columns = ['Mkt-RF','SMB','HML','RF']
        factors_monthly.index = factors_monthly.index+pd.tseries.offsets.MonthEnd(0)
        return factors_monthly