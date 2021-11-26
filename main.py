import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pykalman import KalmanFilter
import DCC
import ARIMA
import stock_data_preprocessor as sdp
from scipy.optimize import minimize
import time

from covariance_matrix import covariance_matrix

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
expCov = covariance_matrix(expR,beta_cov,beta_mean,factor_cov,factor_preds)

# parameters
lb = 0 #0.0
ub = 1 #1.0
useWeekly = False
alpha = 0.1
riskMeasure = 'vol' #vol, VaR, CVaR

def MV(w, cov_mat):   # w = alpha which is the weight, cov_matrix should be known
    return np.dot(w,np.dot(cov_mat,w.T))

n = len(expCov.columns)
muRange = np.arange(0.0055,0.013,0.0002)
volRange = np.zeros(len(muRange))
R = expR
omega = expCov.cov()

wgt = {}   # weight

for i in range(len(muRange)):
    mu = muRange[i]
    wgt[mu] = []
    x_0 = np.ones(n)/ n  #initial guess
    bndsa = ((lb,ub),)
    for j in range(1,n):
        bndsa = bndsa + ((lb,ub),)  #put bound of each stock
    consTR = ({'type':'eq','fun': lambda x: 1 - np.sum(x)},    # 'eq' means equal to; 'fun' is function; this line is meaning that 1-np.sum(x)=0
              {'type':'eq','fun': lambda x: mu - np.dot(x,R)})
    w = minimize(MV, x_0, method = 'SLSQP',constraints=consTR,bounds=bndsa, args=(omega))
    # omega is referring to the cov_mat
    # args means the extra arguments passed to the objective function (就是objective function裏面需要的其他參數）
    # args can be more than one!!!!!!
    volRange[i] = np.dot(w.x,np.dot(omega, w.x.T)) ** 0.5   # w.x是因爲w算出了很多東西，但我們取得是x，其他的還有jac

    wgt[mu].extend(np.squeeze(w.x))

sharpe = np.array([])

for i in range(len(muRange)):
    sharpe.append(muRange[i]/volRange[i])

bestWgt = wgt[muRange[sharpe.argmax()]]
weights.loc[-1]=bestWgt

print("--- %s seconds ---" % (time.time() - start_time))