import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pykalman import KalmanFilter
import DCC
import ARIMA
import stock_data_preprocessor as sdp

import time
start_time = time.time()

# sdp.data_download()
end = datetime.date.today()
start = end - datetime.timedelta(weeks=105)
weekend_date = pd.date_range(start=start, end=end, freq='W-FRI').date
all_price = sdp.data_preprocess()
weights = pd.DataFrame(index=weekend_date, columns=all_price.columns)

for date in weekend_date:
    period_price = all_price[date - datetime.timedelta(weeks=260):date]
    period_price.dropna(how='any', axis=1, inplace=True)
    period_return = period_price.pct_change().iloc[1:]
    
    factors = pd.DataFrame()
    data_array = period_return.to_numpy()
    pca = PCA(n_components=0.8)  # explain 90% data
    pca.fit(data_array)
    print(pca.explained_variance_ratio_)  # the ratio of data explained by PCA vectors
    eigenvectors = pca.components_  # eigenvectors
    j = 0
    for eigenvec in eigenvectors:
        factors[j] = np.dot(period_return, eigenvec)
        j += 1
    print(factors)  # the pca vectors
    
    arima = ARIMA.ARIMA()
    
    for idv_return in period_return:
        kf = KalmanFilter(transition_matrices=np.identity(factors.shape[1] + 1),
                          observation_matrices=np.concatenate((np.ones((factors.shape[0], 1)), factors.to_numpy()), axis=1),
                          transition_offsets=np.zeros(factors.shape[1] + 1),
                          observation_offsets=np.array([0]),
                          em_vars=['transition_covariance',
                                   'observation_covariance', 
                                   'initial_state_mean', 
                                   'initial_state_covariance'],
                          n_dim_state=factors.shape[1] + 1)
    
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    