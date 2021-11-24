# Import libraries
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# Load the data
data = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)
# Date pre-processing
data['Month']=pd.to_datetime(data['Month'], format='%Y-%m-%d')
data.set_index(['Month'], inplace=True)

# Define the d and q parameters to take any value between 0 and 1
q = d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

train_data = data['1949-01-01':'1959-12-01']
# test_data = data['1960-01-01':'1960-12-01']
warnings.filterwarnings("ignore") # specify to ignore warning messages
AICList=[]
SARIMAX_model_list=[]
for i in range(len(test_data.columns)):
    train_data_temp=train_data.iloc[:,i]
    AIC = []
    SARIMAX_model = []
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train_data_temp,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                AIC.append(results.aic)
                SARIMAX_model.append([param, param_seasonal])
            except:
                continue
    AICList.append(AIC)
    SARIMAX_model_list.append(SARIMAX_model)
# AICdf=pd.DataFrame(AICList).transpose()
# AICdf.columns=train_data.columns
# SARIMAXdf=pd.DataFrame(SARIMAX_model_list).transpose()
# SARIMAXdf.columns=train_data.columns

# print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
# Fit this model
predList=[]
for i in range(len(test_data.columns)):
    train_data_temp=train_data.iloc[:,i]
    SARIMAX_model_temp=SARIMAX_model_list[i]
    AIC_temp=AICList[i]
    mod = sm.tsa.statespace.SARIMAX(train_data_temp,
                                    order=SARIMAX_model_temp[AIC_temp.index(min(AIC_temp))][0],
                                    seasonal_order=SARIMAX_model_temp[AIC_temp.index(min(AIC_temp))][1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    # pred = results.get_prediction(start='1958-01-01', dynamic=False) # 1-step ahead forecast
    # pred = results.get_prediction(start='1958-01-01', dynamic=True) # predict last year data
    pred = results.get_forecast('1962-12-01') # forecast
    predList_temp=pred.predicted_mean.values.tolist()
    predList.append(predList_temp)

predDf=pd.DataFrame(predList).transpose()
predDf.columns=train_data.columns
print(predDf)
predDf.to_csv('ARIMA Prediction.csv')

# prediction = pred2.predicted_mean['1960-01-01':'1960-12-01'].values
# # flatten nested list
# truth = list(itertools.chain.from_iterable(test_data.values))
# # Mean Absolute Percentage Error
# MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100
# print('The Mean Absolute Percentage Error for the forecast of year 1960 is {:.2f}%'.format(MAPE))