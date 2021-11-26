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
end = datetime.datetime.date(2021, 11, 19)
start = end - datetime.timedelta(weeks=2)
weekend_date = pd.date_range(start=start, end=end, freq='W-FRI').date
all_price = sdp.data_preprocess()
weights = pd.DataFrame(index=weekend_date, columns=all_price.columns)

for date in weekend_date:
    period_price = all_price[date - datetime.timedelta(weeks=160):date]
    period_price.dropna(how='any', axis=1, inplace=True)
    period_return = period_price.pct_change().iloc[1:]
    
    factors = pd.DataFrame()
    data_array = period_return.to_numpy()
    pca = PCA(n_components=0.8)  # explain 90% data
    pca.fit(data_array)
    # print(pca.explained_variance_ratio_)  # the ratio of data explained by PCA vectors
    eigenvectors = pca.components_  # eigenvectors
    j = 0
    #print("1--- %s seconds ---" % (time.time() - start_time))
    for eigenvec in eigenvectors:
        factors[j] = np.dot(period_return, eigenvec)
        j += 1
    # print(factors)  # the pca vectors
    #print("2--- %s seconds ---" % (time.time() - start_time))
    factor_preds = []
    factor_resids = []
    for i in range(factors.shape[1]):
        factor = factors.iloc[:, i].to_frame()
        arima = ARIMA.ARIMA()
        arima.AICnSARIMAX(factor)
        factor_pred = arima.pred(factor)
        factor_resid = arima.resid(factor)
        factor_preds.append(factor_pred)
        factor_resids.append(factor_resid)
    #print("3--- %s seconds ---" % (time.time() - start_time))
    for idv_return in period_return.T.values:
        transition_matrix = np.identity(factors.shape[1] + 1)
        observation_matrix = np.concatenate((np.ones((factors.shape[0], 1)), factors.to_numpy()), axis=1).reshape(factors.shape[0], 1, factors.shape[1] + 1)
        transition_offset = np.zeros(factors.shape[1] + 1)
        observation_offset = np.array([0])
        kf = KalmanFilter(transition_matrices=transition_matrix,
                          observation_matrices=observation_matrix,
                          transition_offsets=transition_offset,
                          observation_offsets=observation_offset,
                          em_vars=['transition_covariance',
                                   'observation_covariance',
                                   'initial_state_mean',
                                   'initial_state_covariance'],
                          n_dim_state=factors.shape[1] + 1,
                          n_dim_obs=1)
        #kf.em(idv_return, n_iter=5)
        beta_mean, beta_cov = kf.smooth(idv_return)
    #print("4--- %s seconds ---" % (time.time() - start_time))


    dcc = DCC.DCC()
    dccfit = dcc.fit(np.array(factor_resid))
    factor_cov = dccfit.forecast()

    #print("5--- %s seconds ---" % (time.time() - start_time))

factor_preds=[factor_preds[i][0][0] for i in range(len(factor_preds))]
factor_preds.insert(0,1)
expR = np.dot(beta_mean[-1,:],factor_preds)
#TanWeight =
print("--- %s seconds ---" % (time.time() - start_time))