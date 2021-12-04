from __future__ import (absolute_import,division,print_function,
                        unicode_literals)
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from pykalman import KalmanFilter
from arch.univariate import arch_model
import backtrader as bt

import DCC
import ARIMA
import stock_data_preprocessor as sdp
from covariance_matrix import covariance_matrix

end = datetime.date(2021, 10, 31)
start = end + relativedelta(months=-24)
monthend_date = pd.date_range(start=start, end=end, freq='BM').date
all_price = sdp.data_preprocess()
weights = pd.DataFrame(index=monthend_date, columns=all_price.columns)

for date in monthend_date:
    # Data Cleaning
    expected_return = []
    period_price = all_price[date + relativedelta(months=-60):date]
    for ticker in period_price:
        if np.isnan(period_price[ticker].iloc[0]):
            weights.loc[date, ticker] = 0
    period_price.dropna(how='any', axis=1, inplace=True)
    period_return = period_price.pct_change().iloc[1:]

    # PCA
    factors = pd.DataFrame()
    data_array = period_return.to_numpy()
    pca = PCA(n_components=0.8)  # explain 80% variance
    pca.fit(data_array)
    eigenvectors = pca.components_
    j = 0
    for eigenvec in eigenvectors:
        factors[j] = np.dot(period_return, eigenvec)
        j += 1

    # ARMA for 1-period forecast of PCs
    factor_preds = []
    factor_resids = []
    for i in range(factors.shape[1]):
        factor = factors.iloc[:, i].to_frame()
        arima = ARIMA.ARIMA()
        arima.AICnARIMAX(factor)
        factor_pred = arima.pred(factor)
        factor_resid = arima.resid(factor)[0].to_numpy()
        factor_preds.append(factor_pred)
        factor_resids.append(factor_resid)
    factor_resids = np.array(factor_resids)

    # Kalman filter for obtaining betas
    all_beta_past = []
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
        beta_mean, _ = kf.em(idv_return, n_iter=5).smooth(idv_return)
        all_beta_past.append(beta_mean)
        all_beta_mean.append(beta_mean[-1])
        beta_cov = np.cov(beta_mean, rowvar=False)
        all_beta_cov.append(beta_cov)
    all_beta_past, all_beta_mean, all_beta_cov = np.array(all_beta_past), np.array(all_beta_mean), np.array(all_beta_cov)

    # DCC-garch for covariance matrix between PCs
    dcc = DCC.DCC()
    dccfit = dcc.fit(factor_resids)
    factor_cov = dccfit.forecast()

    # Variance for residual of returns
    past_expected_returns = []
    adj_factors = np.insert(factors.to_numpy(), 0, 1, axis=1)
    for i in range(all_beta_past.shape[0]):
        past_expected_return = np.sum(all_beta_past[i] * adj_factors, axis=1)
        past_expected_returns.append(past_expected_return)
    past_expected_returns = np.array(past_expected_returns).T
    return_residual = period_return.to_numpy() - past_expected_returns
    predicted_vars = []
    for i in range(return_residual.shape[1]):
        garch = arch_model(return_residual[:,i], vol='garch', p=1, o=0, q=1)
        garch_fitted = garch.fit(update_freq=0, disp='off')
        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_var = garch_forecast.variance['h.1'].iloc[-1]
        predicted_vars.append(predicted_var)
    predicted_vars = np.array(predicted_vars)

    factor_preds=[factor_preds[i][0][0] for i in range(len(factor_preds))]
    factor_preds.insert(0,1)
    expR = np.dot(all_beta_mean, factor_preds)
    expCov = covariance_matrix(expR, all_beta_cov, all_beta_mean, factor_cov, factor_preds[1:], predicted_vars)

    lb = 0
    ub = 1
    alpha = 0.1

    def MV(w, cov_mat):
        return np.dot(w,np.dot(cov_mat,w.T))

    n = len(expCov.columns)
    muRange = np.arange(0.02,0.09,0.002)
    volRange = np.zeros(len(muRange))
    R = expR
    omega = expCov.cov()
    
    wgt = {}
    
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

# Backtesting
class highest_sharpe_ratio(bt.Strategy):
    
    def __init__(self):
        pass
    
    def next(self):
        today = self.data.datetime.date()
        print(today)
        
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
            print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

dummy_df = pd.read_csv('stock_data1/MMM.csv',
                       index_col='Date',
                       parse_dates=True)[start:end]
dummy_df.loc[:,:] = 0

# 1.creating a cerebro
cerebro = bt.Cerebro(stdstats=False)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.broker.set_cash(100000000.0)
    
path1 = 'stock_data1/'
symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
for symbol in symbols.values:
    file_path = path1 + symbol[0] + '.csv'
    price_matrix = pd.read_csv(file_path,
                                index_col='Date',
                                parse_dates=True)[start:end]
    matrix_start = price_matrix.index[0]
    price_matrix = pd.concat([dummy_df[:matrix_start][1:], price_matrix])
    datafeed = bt.feeds.PandasData(dataname=price_matrix,plot=False)
    cerebro.adddata(datafeed,name=symbol[0])

# 3.add strategies
cerebro.addstrategy(highest_sharpe_ratio)
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
cerebro.addanalyzer(bt.analyzers.DrawDown)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    
# SP500.plot()
    
# 4.run
res = cerebro.run()[0]
print('Final Portfolio Value:',cerebro.broker.get_value())
    
sharpe_ratio = res.analyzers.sharperatio.get_analysis()
print('==========Sharpe Ratio==========')
print('SharpeRatio:',sharpe_ratio['sharperatio'])
    
drawdown_data = res.analyzers.drawdown.get_analysis()
print('==========Draw Down==========')
print('Max Drawdown:',drawdown_data['max']['drawdown'])
print('Max Moneydown:',drawdown_data['max']['moneydown'])
    
trading_analyzer = res.analyzers.tradeanalyzer.get_analysis()
print('==========Trade Analysis==========')
print('Total trades',trading_analyzer['total'])
print('won',trading_analyzer['won'])
print('lost',trading_analyzer['lost'])


# 5.plot results
cerebro.plot(style='candle',volume=False)



