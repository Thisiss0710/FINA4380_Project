# 引入python3.X的一些特性，用于兼容python3.X
from __future__ import (absolute_import,division,print_function,
                        unicode_literals)

import backtrader as bt
import datetime
import pandas as pd

# class half_half_balance(bt.Strategy):
    
#     def __init__(self):
#         pass
    
#     def next(self):
#         today = self.data.datetime.date()
#         year,month = today.year,today.month
#         if month==12:
#             this_month_length = (datetime.datetime(year+1,1,1)-datetime.datetime(year,month,1)).days
#         else:
#             this_month_length = (datetime.datetime(year,month+1,1)-datetime.datetime(year,month,1)).days
#         if today.day == this_month_length:  #月底那一天rebalance
#             for column_name in weight.columns:
#                 for i in weight.index:
#                     ratio = weight.loc[i,column_name]
#                     self.order_target_percent(target=ratio,data=column_name)
#             # self.order_target_percent(target=0.45,data='AAL')
#             # self.order_target_percent(target=0.45,data='A')
#             #要留一部分，不应满仓，可供顾客赎回    
  
# weights = pd.read_csv("D:/CUHK/yr4 sem1/FINA380/pythonProject/final project/wgt.csv",index_col = 0)


weights = pd.read_csv("wgt.csv",index_col = 0)


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
        
        print(today,'Portfolio Value: %.2f' % cerebro.broker.getvalue())

if __name__ == '__main__':

    # 1.creating a cerebro
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    cerebro.broker.set_cash(1000000.0)
        
    path1 = 'stock_data1/'
    symbols = pd.read_csv('S&P500_ticker1.csv', usecols=['Symbol'])
    for symbol in symbols.values:
        file_path = path1 + symbol[0] + '.csv'
        price_matrix = pd.read_csv(file_path,
                                    index_col='Date',
                                    parse_dates=True)
        # price_matrix.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volumn':'volume'},inplace=True)
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
    
