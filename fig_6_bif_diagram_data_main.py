# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:14:34 2021

@author: Max
"""

import numpy as np
from numba import njit
import fig_3_bif_diagram_data_defs
import multiprocessing as mp
import pickle
from timeit import default_timer as timer
#%%
temp=[]
res=100
gamma_val=np.linspace(0.05,1,res)
tau1_val=np.linspace(0.15,0.3,res)
gamma,tau1=np.meshgrid(gamma_val,tau1_val)
gamma=gamma.flatten()
tau1=tau1.flatten()

for i in range(res**2):
    temp1=np.array([gamma[i],tau1[i]])
    temp.append(iter(temp1))
#%%

start=timer()
result=[]
if __name__ =='__main__':
    pool=mp.Pool()
    result=pool.starmap(fig_3_bif_diagram_data_defs.burst_finder,(temp))
    pool.close()
    pool.join()

stop=timer()
print(stop-start)

#%%
with open(r'Simulation_data/Figure_3/bif_diagram_data','wb') as fp:
    pickle.dump(result,fp)
