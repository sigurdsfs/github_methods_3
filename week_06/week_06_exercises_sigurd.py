#%%
from os import error
from re import I
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy import arange, random
import scipy as sp


#%%
#Exercise 1.1
features, output, coef = datasets.make_regression(n_samples = 100, n_features= 4, n_informative= 4, n_targets= 1, noise = 3, coef = True)


# %% i)
#X matrix
x = np.arange(0,101)
x_mat = np.vstack((np.ones(len(x)), x)).T

#y matrix
def y_value_with_error(start,n,error_rate):
    y = np.arange(start,(start+n))
    error = random.normal(loc = 0.0, scale = error_rate, size = len(y))
    return(np.add(y,error))

y = y_value_with_error(100,len(x),2.5)
beta_hat = np.linalg.inv(x_mat.T.dot(x_mat)).dot(x_mat.T).dot(y)
y_hat = np.dot(beta_hat, x_mat.T)

# %% ii)
plt.plot(x,y, 'go', x, y_hat, 'b')
plt.show()

# %% Exercise 1.2 i)
y1 = ([3,2,7,6,9])
y2 = ([10,4,2,1,-3])
y3 = ([15,-2,0,0,3])
y_com = np.concatenate((y1,y2,y3))

# %% Model Matrix
x_mat2 = np.zeros( (len(y_com),3) )
x_mat2[0:5,0] = 1 
x_mat2[5:10,1] = 1
x_mat2[10:15,2] = 1

#Beta /mean calc
beta_hat2 = np.linalg.inv(x_mat2.T.dot(x_mat2)).dot(x_mat2.T).dot(y_com)
print(beta_hat2)

# %% Exercise 1.2 ii)
tm1 = np.array([-1,1,0])
tm2 = np.array([0,1,0])
tm3 = np.array([-1,0,1])
tm = np.vstack((tm1,tm2,tm3))

beta_catagorial = np.dot(tm,beta_hat2)
print(pd.DataFrame({'Name':["beta1","beta2","beta3"], 
                 'Beta Value ii': beta_catagorial,
                 'Beta Balue i': beta_hat2}))

# %% Exercise 1.3 i) 
A_brac = (sum(y1)**2 + sum(y2)**2 + sum(y3)**2)/5
# %% Y and T bracket calculation 
squared_y1 = [number ** 2 for number in y1]
squared_y2 = [number ** 2 for number in y2]
squared_y3 = [number ** 2 for number in y3]
Y_brac = sum(squared_y1 + squared_y2 + squared_y3)
T_brac = sum(y1 + y2 + y3)**2/len(y_com)
# %% 
SS_between_grp, df_between  = A_brac - T_brac, 2 
SS_within_grp, df_within  = Y_brac - A_brac, 3*(len(y1)-1)

ms_between = SS_between_grp/df_between
ms_within = SS_within_grp/df_within
F = ms_between/ms_within #Getting the F-value. 

#Calc p-value. 
alpha = 0.05
p_value = 1 - sp.stats.f.cdf(F,df_between, df_within)
#Check significance
if p_value < alpha:
    print("Significant Result")
else:
    print("Not Significant Result")

# %%
from scipy.stats import f
f.pdf(np.array([0,4]),df_between, df_within)



# %%

