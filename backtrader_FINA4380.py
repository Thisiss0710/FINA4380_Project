# 引入python3.X的一些特性，用于兼容python3.X
from __future__ import (absolute_import,division,print_function,
                        unicode_literals)

import backtrader as bt
import datetime
import pandas as pd

class half_half_balance(bt.Strategy):
    
    def __init__(self):
        pass
    
    def next(self):
        today = self.data.datetime.date()
        year,month = today.year,today.month
        if month==12:
            this_month_length = (datetime.datetime(year+1,1,1)-datetime.datetime(year,month,1)).days
        else:
            this_month_length = (datetime.datetime(year,month+1,1)-datetime.datetime(year,month,1)).days
        if today.day == this_month_length:  #月底那一天rebalance
            self.order_target_percent(target=0.45,data='AAL')
            self.order_target_percent(target=0.45,data='A')
            #要留一部分，不应满仓，可供顾客赎回    
    
if __name__ == '__main__':

    # 1.creating a cerebro
    cerebro = bt.Cerebro(stdstats=False)
    #可以把默认的obeserver在图上的表示线关掉
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)

    # 2.add data feeds
    # create a data feed
    total_df = pd.read_csv('D:/CUHK/yr4 sem1/FINA380/project_data/collected_adj_close.csv',index_col='Date',parse_dates=True)
    # total_df = pd.read_csv('D:/CUHK/yr4 sem1/FINA380/project_data/A_AAL.csv',index_col='Date',parse_dates=True)
    
    #add the data feed to cerebro
    for col_name in total_df.columns:
        dataframe = total_df[[col_name]]
        for col in ['open','high','low','close']:
            dataframe[col] = dataframe[col_name]
        dataframe['volume']=10000000000000000000
        datafeed = bt.feeds.PandasData(dataname=dataframe,plot=False)
        cerebro.adddata(datafeed,name=col_name)

    # 3.add strategies
    cerebro.addstrategy(half_half_balance)

    # 4.run
    cerebro.run()
    print('value:',cerebro.broker.get_value())

    
    # 5.plot results
    cerebro.plot(style='candle',volume=False)



