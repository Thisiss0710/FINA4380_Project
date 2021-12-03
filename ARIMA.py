# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class ARIMA():
    def __init__(self):
        self.AICList=[]
        self.ARIMAX_model_list=[]
        self.predList=[]
        # Define the d and q parameters to take any value between 0 and 1
        q = range(0, 2)
        d = range(0, 1)
        # Define the p parameters to take any value between 0 and 3
        p = range(0, 4)
        # Generate all different combinations of p, q and q triplets
        self.pdq = list(itertools.product(p, d, q))

    def AICnARIMAX(self,train):
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        self.AICList=[]
        self.ARIMAX_model_list=[]
        for i in range(len(train.columns)):
            train_data_temp=train.iloc[:,i]
            AIC = []
            ARIMAX_model = []
            for param in self.pdq:
                    try:
                        mod = sm.tsa.arima.ARIMA(train_data_temp,
                                                       order=param,
                                                       enforce_stationarity=False,
                                                       enforce_invertibility=False)

                        results = mod.fit() #HERE
                        AIC.append(results.aic)
                        ARIMAX_model.append([param])
                    except:
                        print("Error")
            self.AICList.append(AIC)
            self.ARIMAX_model_list.append(ARIMAX_model)
        # AICdf=pd.DataFrame(AICList).transpose()
        # AICdf.columns=train_data.columns
        # ARIMAXdf=pd.DataFrame(ARIMAX_model_list).transpose()
        # ARIMAXdf.columns=train_data.columns
        # print('The smallest AIC is {} for model ARIMAX{}x{}'.format(min(AIC), ARIMAX_model[AIC.index(min(AIC))][0],ARIMAX_model[AIC.index(min(AIC))][1]))

# Fit this model
    def pred(self,train):
        self.predList=[]
        for i in range(len(train.columns)):
            train_data_temp = train.iloc[:, i].to_frame()
            ARIMAX_model_temp=self.ARIMAX_model_list[i]
            AIC_temp=self.AICList[i]
            mod = sm.tsa.arima.ARIMA(train_data_temp,
                                           order=ARIMAX_model_temp[AIC_temp.index(min(AIC_temp))][0],
                                           enforce_stationarity=True,
                                           enforce_invertibility=False)
            results = mod.fit()
            pred = results.get_prediction(start=-1,dynamic=False) # 1-step ahead forecast
            # pred = results.get_prediction(start='1958-01-01', dynamic=True) # predict last year data
            # pred = results.get_forecast(ForecastTillDate) # forecast
            predList_temp=pred.predicted_mean.values.tolist()
            self.predList.append(predList_temp)
            # predDf=pd.DataFrame(self.predList).transpose()
            # predDf.columns=train.columns
            # predDf.to_csv('prediction.csv')
        return self.predList

    def resid(self,train):
        self.predList=[]
        for i in range(len(train.columns)):
            train_data_temp = train.iloc[:, i]
            ARIMAX_model_temp=self.ARIMAX_model_list[i]
            AIC_temp=self.AICList[i]
            mod = sm.tsa.arima.ARIMA(train_data_temp,
                                           order=ARIMAX_model_temp[AIC_temp.index(min(AIC_temp))][0],
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
            results = mod.fit()
            pred = results.resid # Get residual value
            # predList_temp=pred.predicted_mean.values.tolist()
            self.predList.append(pred)
        return self.predList
            # predDf=pd.DataFrame(self.predList).transpose()
            # predDf.columns=train.columns
            # predDf.to_csv('residual.csv')

# prediction = pred2.predicted_mean['1960-01-01':'1960-12-01'].values
# # flatten nested list
# truth = list(itertools.chain.from_iterable(test_data.values))
# # Mean Absolute Percentage Error
# MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100
# print('The Mean Absolute Percentage Error for the forecast of year 1960 is {:.2f}%'.format(MAPE))

