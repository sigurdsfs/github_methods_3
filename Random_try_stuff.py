#%%
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
#%% 
def plot_stuff(x):
    y = np.zeros(shape = x.shape)
    int = -105.570 
    beta = 2.540
    y = int + x * beta
    return y
# %%
x = np.arange(-10,100,0.1)
y = plot_stuff(x)

plt.plot(x,y,"-r")
plt.xlabel("shoesize")
plt.ylabel("log-odds")
plt.title("Showing that log-odds are unbounded")
plt.show()

# %%

# %%
from scipy.special import logit, expit
#function for inverse sigmoid transformation.
def plot_stuff_prop_scale(x):
    y = np.zeros(shape = x.shape)
    int = -105.570 
    beta = 2.540
    y = expit(int) + expit(x * beta)
    return y

# %%
#Caclulate
x_prop = np.arange(-10,10,0.1)
y_prop = plot_stuff_prop_scale(x_prop)

#Plot
plt.plot(x_prop,y_prop,"-r")
plt.xlabel("shoesize")
plt.ylabel("Propability")
plt.title("After sigmoid transformation we get %")
plt.show()

# %%
