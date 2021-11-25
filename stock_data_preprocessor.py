import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from sklearn.decomposition import PCA

def data_download(date=datetime.date(2013, 1, 1)):
    # import data
    start = date
    end = datetime.date.today()
    modified_start = start + datetime.timedelta(days=1)
    modified_end = end + datetime.timedelta(days=1)
    
    symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
    for symbol in symbols.values:
        try:
            stock = yf.download(symbol[0],start=modified_start, end=modified_end)
            stock.to_csv(f'stock_data/{symbol[0]}.csv')
        except Exception:
            continue
        
# close the adj_close prices into a single matrix 'price_matrix'
def data_preprocess(date):
    price_matrix = pd.DataFrame()
    path = 'stock_data/'
    symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
    for symbol in symbols.values:
        file_path = path + symbol[0] + '.csv'
        adj_close = pd.read_csv(file_path,
                              index_col='Date',
                              usecols=['Date','Adj Close'],
                              parse_dates=True)
        if adj_close.index[0].date() < datetime.date(2019, 1, 1):
            adj_close.rename(columns={'Adj Close':symbol[0]},inplace=True)
            if price_matrix.empty:
                price_matrix = adj_close
            else:
                price_matrix = pd.merge(price_matrix, adj_close, on=['Date'], how='outer')

    # price_matrix.set_index(price_matrix.columns[0],inplace=True)
    # price_matrix.index = pd.to_datetime(price_matrix.index,format="%Y/%m/%d")
    price_matrix.interpolate(method='spline', order=3, inplace=True)
    price_matrix.sort_index(inplace=True)
    print(price_matrix.index)
    price_matrix.to_csv('collected_adj_close.csv')
    
    # process data and fill the blanks
    grouped_price_matrix = price_matrix.groupby(pd.Grouper(freq='MS')).nth(0)
    # grouped_price_matrix.index = pd.to_datetime(grouped_price_matrix.index,format="%Y/%m/%d")
    
    grouped_price_matrix.dropna(axis='columns',inplace=True)
    price_matrix.to_csv('collected_adj_close1.csv')

data_preprocess('c')