import pandas as pd
import yfinance as yf
import datetime
import numpy as np
from sklearn.decomposition import PCA

# import data
start = datetime.date(2019,1,1)
end = datetime.date.today()
modified_start = start + datetime.timedelta(days=1)
modified_end = end + datetime.timedelta(days=1)

symbols = pd.read_csv('D:/CUHK/yr4 sem1/FINA380/S&P500_ticker1.csv', usecols=['Symbol'])
for symbol in symbols.values:
    try:
        stock = yf.download(symbol[0],start = modified_start, end = modified_end)
        stock.to_csv(f'D:/CUHK/yr4 sem1/FINA380/project_data/{symbol[0]}.csv')
    except Exception:
        print('Download failed. Check if there is any error.')
        
# close the adj_close prices into a single matrix 'price_matrix'
price_matrix = pd.DataFrame()
path = 'D:/CUHK/yr4 sem1/FINA380/project_data/'
i=0
for symbol in symbols.values:
    file_path = path + symbol[0] + '.csv'
    # file_data=pd.read_csv(file_path)
    adj_close=pd.read_csv(file_path,usecols=['Date','Adj Close'])
    # adj_close.set_index(keys=file_data['Date'],inplace=True)
    adj_close.rename(columns={'Adj Close':symbol[0]},inplace=True)
    if i == 0:
        price_matrix = adj_close
    else:
        price_matrix=pd.merge(price_matrix,adj_close,on=['Date'],how='outer')
    i+=1
price_matrix.set_index(price_matrix.columns[0],inplace=True)
price_matrix.to_csv(f'D:/CUHK/yr4 sem1/FINA380/project_data/collected_adj_close.csv')

#process data and fill the blanks
# print(np.where(np.isnan(price_matrix)))
# print(price_matrix.index[np.where(np.isnan(price_matrix))[0]])
# price_matrix.fillna(method='ffill',inplace=True)
price_matrix.dropna(inplace=True)
# price_matrix.to_csv(f'D:/CUHK/yr4 sem1/FINA380/project_data/collected_adj_close1.csv')

# PCA
factors=pd.DataFrame()
data_array = price_matrix.to_numpy()
pca = PCA(n_components=0.9)  # explain 90% data
pca.fit(data_array)
print(pca.explained_variance_ratio_)  # the ratio of data explained by PCA vectors
eigenvectors = pca.components_  # eigenvectors
j = 1
for eigenvec in eigenvectors:
    factors[j] = np.dot(price_matrix,eigenvec)
    j+=1
print(factors)  # the pca vectors









