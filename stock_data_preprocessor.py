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
            
    SP500 = yf.download('SPY',start=datetime.date(2019, 10, 2), end=modified_end)
    SP500.to_csv(f'stock_data/SP500.csv')
    
    
# Put the adj_close prices into a single matrix 'price_matrix'
def data_preprocess():
    price_matrix = pd.DataFrame()
    path = 'stock_data/'
    symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
    for symbol in symbols.values[:]:
        file_path = path + symbol[0] + '.csv'
        adj_close = pd.read_csv(file_path,
                                index_col='Date',
                                usecols=['Date','Adj Close'],
                                parse_dates=True)
        adj_close.rename(columns={'Adj Close':symbol[0]},inplace=True)
        if price_matrix.empty:
            price_matrix = adj_close
        else:
            price_matrix = pd.merge(price_matrix, adj_close, on=['Date'], how='outer')

    price_matrix.interpolate(method='spline', order=3, inplace=True)
    price_matrix.sort_index(inplace=True)

    grouped_price_matrix = price_matrix.groupby(pd.Grouper(freq='M')).nth(-1)     # process data and fill the blanks
    
    return grouped_price_matrix

    
    # process data and fill the blanks
    grouped_price_matrix = price_matrix.groupby(pd.Grouper(freq='M')).nth(-1)
    return grouped_price_matrix



# create data files without adj close
def data_adjustment():
    price_matrix = pd.DataFrame()
    path = 'stock_data/'
    symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
    for symbol in symbols.values:
        file_path = path + symbol[0] + '.csv'
        price_matrix = pd.read_csv(file_path,
                                   index_col='Date',
                                   usecols=['Date','Open','High','Low','Close','Volume'],
                                   parse_dates=True)

        price_matrix.interpolate(method='spline', order=3, inplace=True)
        price_matrix.sort_index(inplace=True)

        price_matrix = price_matrix.groupby(pd.Grouper(freq='BM')).nth(-1)
        price_matrix.to_csv(f'stock_data1/{symbol[0]}.csv')

def SP500():
    price_matrix = pd.read_csv('stock_data/SP500.csv',
                                index_col='Date',
                                usecols=['Date','Adj Close'],
                                parse_dates=True)
    price_matrix.rename(columns={'Adj Close':'SP500'},inplace=True)

    price_matrix.interpolate(method='spline', order=3, inplace=True)
    price_matrix.sort_index(inplace=True)
    price_matrix = price_matrix.groupby(pd.Grouper(freq='M')).nth(-1)
    price_matrix.to_csv(f'stock_data1/SP500.csv')

    price_matrix = price_matrix.groupby(pd.Grouper(freq='BM')).nth(-1)

