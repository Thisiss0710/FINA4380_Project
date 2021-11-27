from __future__ import (absolute_import,division,print_function,
                        unicode_literals)
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from pykalman import KalmanFilter
import DCC
import ARIMA
import stock_data_preprocessor as sdp

import time
from dateutil.relativedelta import relativedelta
import backtrader as bt
from covariance_matrix import covariance_matrix

start_time = time.time()

# sdp.data_download()
end = datetime.date(2021, 10, 31)
start = end + relativedelta(months=-24)
monthend_date = pd.date_range(start=start, end=end, freq='BM').date
all_price = sdp.data_preprocess()
weights = pd.DataFrame(index=monthend_date, columns=all_price.columns)

for date in monthend_date:
    expected_return = []
    period_price = all_price[date + relativedelta(months=-60):date]
    for ticker in period_price:
        if period_price[ticker].iloc[0] == np.nan:
            weights.at[date, ticker] = 0
    period_price.dropna(how='any', axis=1, inplace=True)
    period_return = period_price.pct_change().iloc[1:]
    
    factors = pd.DataFrame()
    data_array = period_return.to_numpy()
    pca = PCA(n_components=0.8)  # explain 80% data
    pca.fit(data_array)
    eigenvectors = pca.components_
    j = 0
    for eigenvec in eigenvectors:
        factors[j] = np.dot(period_return, eigenvec)
        j += 1

    factor_preds = []
    factor_resids = []
    for i in range(factors.shape[1]):
        factor = factors.iloc[:, i].to_frame()
        arima = ARIMA.ARIMA()
        arima.AICnSARIMAX(factor)
        factor_pred = arima.pred(factor)
        factor_resid = arima.resid(factor)[0].to_numpy()
        factor_preds.append(factor_pred)
        factor_resids.append(factor_resid)
    factor_resids = np.array(factor_resids)
        
    all_beta_mean = []
    all_beta_cov = []
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
        beta_mean, beta_cov = kf.smooth(idv_return)
        all_beta_mean.append(beta_mean)
        all_beta_cov.append(beta_cov)
    all_beta_mean, all_beta_cov = np.array(all_beta_mean), np.array(all_beta_cov)

    dcc = DCC.DCC()
    dccfit = dcc.fit(factor_resids)
    factor_cov = dccfit.forecast()

    factor_preds=[factor_preds[i][0][0] for i in range(len(factor_preds))]
    factor_preds.insert(0,1)
    expR = np.dot(all_beta_mean[:,-1,:],factor_preds)
    expCov = covariance_matrix(expR, all_beta_cov[:,-1,:,:], all_beta_mean[:,-1,:], factor_cov, factor_preds[1:])

lb = 0
ub = 1
alpha = 0.1

def MV(w, cov_mat):
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
    x_0 = np.ones(n)/ n 
    bndsa = ((lb,ub),)
    for j in range(1,n):
        bndsa = bndsa + ((lb,ub),) 
    consTR = ({'type':'eq','fun': lambda x: 1 - np.sum(x)},
              {'type':'eq','fun': lambda x: mu - np.dot(x,R)})
    w = minimize(MV, x_0, method = 'SLSQP', constraints=consTR,bounds=bndsa, args=(omega), options={'disp': False})
    volRange[i] = np.dot(w.x,np.dot(omega, w.x.T)) ** 0.5

    wgt[mu].extend(np.squeeze(w.x))

sharpe = np.array([])

for i in range(len(muRange)):
    sharpe = np.append(sharpe, muRange[i]/volRange[i])

bestWgt = wgt[muRange[sharpe.argmax()]]
nulls = pd.isnull(weights.loc[date])
y = [i for i in range(len(nulls)) if nulls.iloc[i] == True]
for i in range(len(y)):
    weights.loc[date].iloc[y[i]] = bestWgt[i]

class highest_sharpe_ratio(bt.Strategy):
    
    def __init__(self):
        pass
    
    def next(self):
        today = self.data.datetime.date()
        year,month = today.year,today.month
        if month==12:
            this_month_length = (datetime.datetime(year+1,1,1)-datetime.datetime(year,month,1)).days
        else:
            this_month_length = (datetime.datetime(year,month+1,1)-datetime.datetime(year,month,1)).days
        if today.day == this_month_length:
            for column_name in weights.columns:
                for i in weights.index:
                    ratio = weights.loc[i,column_name]
                    self.order_target_percent(target=ratio,data=column_name)

# 1.creating a cerebro
cerebro = bt.Cerebro(stdstats=False)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
    
path1 = 'stock_data1/'
symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
for symbol in symbols.values:
    file_path = path1 + symbol[0] + '.csv'
    price_matrix = pd.read_csv(file_path,
                               index_col='Date',
                               parse_dates=True)
    price_matrix.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volumn':'volume'},inplace=True)
    datafeed = bt.feeds.PandasData(dataname=price_matrix,plot=False)
    cerebro.adddata(datafeed,name=symbol[0])

# 3.add strategies
cerebro.addstrategy(highest_sharpe_ratio)
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
cerebro.addanalyzer(bt.analyzers.DrawDown)

# 4.run
res = cerebro.run()[0]
print('value:',cerebro.broker.get_value())
print('SharpeRatio:',res.analyzers.sharperatio.get_analysis())
print('DrawDown:',res.analyzers.drawdown.get_analysis())
    
# 5.plot results
cerebro.plot(style='candle',volume=False)




