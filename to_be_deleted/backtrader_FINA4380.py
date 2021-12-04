# 引入python3.X的一些特性，用于兼容python3.X
from __future__ import (absolute_import,division,print_function,
                        unicode_literals)

import backtrader as bt
import datetime
import pandas as pd



class highest_sharpe_ratio(bt.Strategy):
    
    def __init__(self):
        today = self.data.datetime.date()
        self.weights = pd.read_csv('final_weights.csv',index_col='Date',parse_dates=True)
        self.i = 0
        for column_name in self.weights.columns:
            ratio = self.weights[column_name].iloc[self.i]
            self.order_target_percent(target=ratio,data=column_name)
        print(today,'Portfolio Value: %.2f' % cerebro.broker.getvalue())
        
        self.portfolio_value = pd.DataFrame()

    def next(self):        
        today = self.data.datetime.date()    
        if self.i < 24:
            self.i=self.i+1
        #print(self.i)
        for column_name in self.weights.columns:
            ratio = self.weights[column_name].iloc[self.i]
            self.order_target_percent(target=ratio,data=column_name)

        print(today,'Portfolio Value: %.2f' % cerebro.broker.getvalue())

        #self.portfolio_value.append(cerebro.broker.getvalue())
        #self.portfolio_value.to_csv(f'stock_data1/portfolio_value.csv')
        
        self.portfolio_value[today]=cerebro.broker.getvalue()
        self.portfolio_value.to_csv(f'stock_data1/portfolio_value.csv')

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
    
