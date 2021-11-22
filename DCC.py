import numpy as np
import scipy.optimize as spo
from arch.univariate import arch_model
from math import log, pi
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import pandas as pd

class DCC():
    '''
    DCC-GARCH model.
    This model assumes the variables follow normal distribution unconditionally.
    
    Parameters:
        theta: array-like, default=[0.69, 0.29]
            Initial guess for theta_1 and theta_2 for DCC model.
        method: str, default='SLSQP'
            Method for parameter optimization.
            
    Methods:
        theta_(): Return initial guess for theta_1 and theta_2
        set_theta(theta): Update initial guess for theta_1 and theta_2
        fit(train): Fit DCC_GARCH model
    '''
    def __init__(self, theta=[0.69, 0.29], method='SLSQP'):
        self.theta = np.array(theta)    # Parameters for DCC model
        self.method = method    # Method for optimization
    
    def theta_(self):
        return self.theta
    
    def set_theta(self, theta):
        self.theta = np.array(theta)
          
    def fit(self, train):
        def Q_uncon(y):
            train = np.array(y)
            return np.corrcoef(train)

        def Q(theta1, theta2, y):
            y = np.array(y)
            eta = y.T
            eta = (eta - np.mean(eta, axis=0)) / np.std(eta, axis=0)
            eta = eta.T   
            Q_int = Q_uncon(eta)
            Q_list = [Q_int]
            t = eta.shape[1] - 1
            for i in range(t):
                etat_1 = eta[:, -i]
                Qt_1 = Q_list[0]
                Qt = (1 - theta1 - theta2) * Q_int + theta1 * Qt_1 + theta2 * np.outer(etat_1, etat_1)
                Q_list = [Qt] + Q_list
            return Q_list

        def rho(theta1, theta2, y):
            Qt = Q(theta1, theta2, y)
            rho_list = []
            n = Qt[0].shape[1]
            for i in Qt:
                Jt = np.eye(n) / np.sqrt(np.abs(i))
                Rt = np.dot(np.dot(Jt, i), Jt)
                rho_list.append(Rt)
            return np.array(rho_list)

        def cov(theta1, theta2, y):
            corr = rho(theta1, theta2, y)
            std = np.zeros(y.shape)
            for i in range(y.shape[0]):
                garch = arch_model(y[i], vol='garch', p=1, o=0, q=1)
                garch_fitted = garch.fit(update_freq=0, disp='off')
                std_hist = garch_fitted.conditional_volatility
                std[i,:] = std_hist
            for t in range(corr.shape[0]):
                for i in range(corr.shape[1]):
                    for j in range(corr.shape[2]):
                        corr[t, i, j] = corr[t, i, j] * std[i, t] * std[j, t]
            return corr

        def negative_likelihood(params, train):
            theta1 = params[0]
            theta2 = params[1]
            sigma = cov(theta1, theta2, train)
            t = sigma.shape[0]
            k = sigma.shape[1]
            log_likelihood = 0
            for i in range(t):
                a = train[:,i]
                s = sigma[i]
                #log_likelihood += log(gamma((v + k) / 2)) - k / 2 * log(pi * (v - 2)) - log(gamma(v / 2)) - 1 / 2 * log(np.linalg.det(s)) - (v + k) / 2 * log(1 + 1 / (v - 2) * np.dot(np.dot(a.T, np.linalg.inv(s)), a))
                log_likelihood += -k / 2 * log(2 * pi) - 1 / 2 * log(np.linalg.det(s)) - 1 / 2 * np.dot(np.dot(a.T, np.linalg.inv(s)), a)
            return -log_likelihood
        
        theta_int = self.theta
        res = spo.minimize(negative_likelihood,
                            (theta_int[0], theta_int[1]),
                            args=(train),
                            method=self.method,
                            bounds=((0, 1), (0, 1)),
                            constraints=({'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}),
                            options={'disp': False})
        print(res)
        return DCCfit(train, res.x)
        
class DCCfit():
    '''
    Fit DCC-GARCH model.
    
    Parameters:
        y: array_like of shape (n_features, n_samples)
            Features to be fitted.
        theta: array-like, default=[0.69, 0.29]
            Fitted values of theta_1 and theta_2 for DCC model.
            
    Methods:
        Q_uncon(): Return unconditional correlation matrix
        Q(); Return Q_t for t=0,...,T
        rho(): Return conditional correlation matrice for t=0,...,T
        cov(): Return conditional variance-covariance matrice for t=0,...,T
        forecast(): Return 1-period forecast of variance-covariance matrix
    '''
    def __init__(self, y, theta):
        self.y = np.array(y)
        self.theta = theta
        eta = self.y.T
        eta = (eta - np.mean(eta, axis=0)) / np.std(eta, axis=0)
        self.eta = eta.T
        factor_garch = []
        for i in range(y.shape[0]):
            garch = arch_model(y[i], vol='garch', p=1, o=0, q=1, dist='studentsT')
            garch_fitted = garch.fit(update_freq=0, disp='off')
            factor_garch.append(garch_fitted)
        self.y_garch = np.array(factor_garch)
    
    def Q_uncon(self):
        if type(self.eta) != 'numpy.ndarray':
            train = np.array(self.eta)
        return np.corrcoef(train)
    
    def Q(self):
        eta = self.eta  
        Q_int = self.Q_uncon()
        Q_list = [Q_int]
        theta1 = self.theta[0]
        theta2 = self.theta[1]
        t = eta.shape[1] - 1
        for i in range(t):  
            etat_1 = eta[:, -i]
            Qt_1 = Q_list[0]
            Qt = (1 - theta1 - theta2) * Q_int + theta1 * Qt_1 + theta2 * np.outer(etat_1, etat_1)
            Q_list = [Qt] + Q_list
        return np.array(Q_list)

    def rho(self):
        Qt = self.Q()
        rho_list = []
        n = Qt[0].shape[1]
        for i in Qt:
            Jt = np.eye(n) / np.sqrt(np.abs(i))
            Rt = np.dot(np.dot(Jt, i), Jt)
            rho_list.append(Rt)
        return np.array(rho_list)
    
    def cov(self):
        yt = self.y
        corr = self.rho()
        std = np.zeros(yt.shape)
        for i in range(yt.shape[0]):
            garch_fitted = self.y_garch[i]
            std_hist = garch_fitted.conditional_volatility
            std[i,:] = std_hist
        for t in range(corr.shape[0]):
            for i in range(corr.shape[1]):
                for j in range(corr.shape[2]):
                    corr[t, i, j] = corr[t, i, j] * std[i, t] * std[j, t]
        return corr
    
    def forecast(self):
        var_forecast = []
        for i in self.y_garch:
            garch_forecast = i.forecast(horizon=1)
            predicted_var = garch_forecast.variance['h.1'].iloc[-1]
            var_forecast.append(predicted_var)
        theta1, theta2 = self.theta[0], self.theta[1]
        previous_Q, previous_eta = self.Q()[-1], self.eta[:, -1]
        rho_forecast = (1 - theta1 - theta2) * self.Q_uncon() + theta1 * previous_Q + theta2 * np.outer(previous_eta, previous_eta)
        cov_forecast = np.zeros((rho_forecast.shape[0], rho_forecast.shape[0]))
        for i in range(rho_forecast.shape[0]):
            for j in range(rho_forecast.shape[1]):
                cov_forecast[i, j] = rho_forecast[i, j] * np.sqrt(var_forecast[i]) * np.sqrt(var_forecast[j])
        return cov_forecast
    
## for testing purpose only    
# sigmat = [np.random.normal(loc=0, scale=1) ** 2]
# epsilont = [np.random.normal(loc=0, scale=1)]
# at = [np.sqrt(sigmat[-1]) * epsilont[-1]]
# alpha = 0.7
# beta = 0.29
# for i in range(1100):
#     sigmat.append(1 + alpha * at[-1] ** 2 + beta * sigmat[-1])
#     epsilont.append(np.random.normal(loc=0, scale=1))
#     at.append(np.sqrt(sigmat[-1]) * epsilont[-1])

# at = np.array(at) + 1
# sigmat = sigmat[101:]
# epsilont = epsilont[101:]
# at = at[101:]
# bt = at + np.random.normal(0, 2, 1000)
# ct = bt + np.random.gamma(2, size=1000)
# dcc = DCC()
# dcc_fit = dcc.fit(np.stack((at, bt, ct)))
# q = dcc_fit.Q()
# r = dcc_fit.rho()
# c = dcc_fit.cov()

# plt.plot(r[:,0,1])

# print(dcc_fit.forecast())


start = datetime.date(2012,1,1)    # e.g. datetime.date(2000,1,1)
end = datetime.date.today()    #e.g. datetime.date.today()
modified_start = start + datetime.timedelta(days=1)
modified_end = end + datetime.timedelta(days=1)

aapl = yf.download('AAPL', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100
tsla = yf.download('TSLA', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100
nvda = yf.download('NVDA', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100
wmt = yf.download('WMT', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100
spy = yf.download('SPY', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100
qqq = yf.download('QQQ', start=modified_start, end=modified_end)['Adj Close'].pct_change().dropna().T * 100


dcc = DCC()
dcc_fit = dcc.fit(np.stack((aapl, tsla, nvda, wmt)))
print(dcc_fit.forecast())

r = dcc_fit.rho()
c = dcc_fit.cov()

plt.plot(c[:,0,0])
plt.plot(c[:,1,1])
plt.plot(c[:,2,2])
plt.plot(c[:,3,3])
plt.show()

plt.plot(r[:,0,1])
plt.plot(r[:,0,2])
plt.plot(r[:,0,3])
plt.plot(r[:,1,2])
plt.plot(r[:,1,3])
plt.plot(r[:,2,3])
plt.legend([1,2,3,4,5,6])
plt.show()
