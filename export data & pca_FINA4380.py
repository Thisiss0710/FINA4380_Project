import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from sklearn.decomposition import PCA

# import data
start = datetime.date(2013,1,1)
end = datetime.date.today()
modified_start = start + datetime.timedelta(days=1)
modified_end = end + datetime.timedelta(days=1)

symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
for symbol in symbols.values:
    try:
        stock = yf.download(symbol[0],start = modified_start, end = modified_end)
        stock.to_csv(f'stock_data/{symbol[0]}.csv')
    except Exception:
        continue
        
# close the adj_close prices into a single matrix 'price_matrix'
price_matrix = pd.DataFrame()
path = 'stock_data/'
i = 0
for symbol in symbols.values:
    file_path = path + symbol[0] + '.csv'
    adj_close=pd.read_csv(file_path,usecols=['Date','Adj Close'])
    if adj_close['Date'][0] == '2019-01-02':
        adj_close.rename(columns={'Adj Close':symbol[0]},inplace=True)
        if i == 0:
            price_matrix = adj_close
        else:
            price_matrix=pd.merge(price_matrix,adj_close,on=['Date'],how='outer')
    
    i += 1
price_matrix.set_index(price_matrix.columns[0],inplace=True)
price_matrix.index = pd.to_datetime(price_matrix.index,format="%Y/%m/%d")
price_matrix.to_csv('collected_adj_close.csv')

# process data and fill the blanks
grouped_price_matrix = price_matrix.groupby(pd.Grouper(freq='W')).tail(1)
grouped_price_matrix.index = pd.to_datetime(grouped_price_matrix.index,format="%Y/%m/%d")
grouped_price_matrix.interpolate(method='cubicspline',axis='columns',inplace=True)
grouped_price_matrix.dropna(axis='columns',inplace=True)
price_matrix.to_csv('collected_adj_close1.csv')

# PCA
factors=pd.DataFrame()
data_array = grouped_price_matrix.to_numpy()
pca = PCA(n_components=0.9)  # explain 90% data
pca.fit(data_array)
print(pca.explained_variance_ratio_)  # the ratio of data explained by PCA vectors
eigenvectors = pca.components_  # eigenvectors
j = 1
for eigenvec in eigenvectors:
    factors[j] = np.dot(grouped_price_matrix,eigenvec)
    j+=1
print(factors)  # the pca vectors
