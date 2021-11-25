from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kalF(X, P, A, Q, z, R):
    """
    :param X: Stock return vector
    :param P: Stock return variance
    :param A: Transition Matrix
    :param Q: Transition Covariance
    :param z: Observation
    :param R: Observation Covariance
    :return: The predicted t+1 return
    """

    kf = KalmanFilter(initial_state_mean=X,  # Initial r (predicted from last state)
                      initial_state_covariance=P,  # Variance of this r
                      transition_matrices=A,  # Prediction model
                      transition_covariance=Q,  # Prediction variance?
                      observation_covariance=R, )  # Gaussian?

    # Get the Kalman smoothing
    state_means, _ = kf.smooth(z)
    output = np.array(state_means)
    return output
