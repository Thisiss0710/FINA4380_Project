import numpy as np
import pandas as pd

# beta_matrix is the beta covariance
# beta_expected_matrix is the expected value of beta
# pca_cov is the pca covariance matrix
# pca_array is the expected value of pca
# s_rt is the array of stock return

def covariance_matrix(date,s_rt,beta_matrix,beta_expected_matrix,pca_cov,pca_array):
    l = len(s_rt)
    whole_cov_matrix = pd.DataFrame(np.zeros((l,l)))
    for m in range(len(s_rt)):  # s_rt = stock_return and stock m 
        for n in range(len(s_rt)):  # stock n
            whole_cov = 0
            if m != n :  
                for x in range(len(pca_array)+1):  # x is the xth beta of stock m
                    if x == 0:
                        single_cov = 0
                        whole_cov = whole_cov + single_cov
                    else:
                        for y in range(len(pca_array)+1):  # y is the yth beta of stock n 
                            if y == 0:
                                single_cov = 0
                                whole_cov = whole_cov + single_cov
                            else:
                                single_cov =  beta_matrix.loc[m,x]*beta_matrix.loc[n,y]*pca_cov.loc[x-1,y-1]
                                whole_cov = whole_cov + single_cov
            if m == n:
                for x in range(len(pca_array)+1):
                    for y in range(len(pca_array)+1):                
                        if x == 0 and y == 0:
                            single_cov = beta_matrix.loc[x,y] 
                            whole_cov = whole_cov + single_cov
                        elif x == 0 and y != 0:
                            single_cov = pca_array[y-1] * beta_matrix.loc[x,y]
                            whole_cov = whole_cov + single_cov
                        else:
                            single_cov = beta_expected_matrix.loc[m,x]*beta_expected_matrix.loc[m,y]*pca_cov.loc[x-1,y-1] + pca_array[x-1]*pca_array[y-1]*beta_matrix.loc[x,y] + pca_cov.loc[x-1,y-1]*beta_matrix.loc[x,y] 
                            whole_cov = whole_cov + single_cov
            whole_cov_matrix[m,n] = whole_cov
    return(whole_cov_matrix)