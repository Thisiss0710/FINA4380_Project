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
class highest_sharpe_ratio(bt.Strategy):
    
    def __init__(self):
        pass
    
    def next(self):
        
        today = self.data.datetime.date()

        year,month = today.year,today.month
        weights=pd.read_csv('wgt.csv',
                            index_col='Date',
                            parse_dates=True)
        if month==12:
            this_month_length = (datetime.datetime(year+1,1,1)-datetime.datetime(year,month,1)).days
        else:
            this_month_length = (datetime.datetime(year,month+1,1)-datetime.datetime(year,month,1)).days

        if today.day == this_month_length:
            for column_name in weights.columns:
                for i in weights.index:
                    ratio = weights.loc[i,column_name]
                    self.order_target_percent(target=ratio,data=column_name)
        
        print(today,'Portfolio Value: %.2f' % cerebro.broker.getvalue())

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
    
# SP500.plot()
    
# 4.run
res = cerebro.run()[0]
print('Final Portfolio Value:',cerebro.broker.get_value())
    
sharpe_ratio = res.analyzers.sharperatio.get_analysis()
print('==========Sharpe Ratio==========')
print('SharpeRatio:',sharpe_ratio['sharperatio'])
    
drawdown_data = res.analyzers.drawdown.get_analysis()
print('==========Sharpe Ratio==========')
print('Max Drawdown:',drawdown_data['max']['drawdown'])
print('Max Moneydown:',drawdown_data['max']['moneydown'])
        
# 5.plot results
cerebro.plot(style='candle',volume=False)
