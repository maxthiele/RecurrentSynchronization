# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:59:15 2021

@author: Max
"""
import numpy as np
from numba import njit
import fig_3_static_grid_defs
import multiprocessing as mp
import pickle

#%% Grid Construction
temp=[]
res1=201
res2=201
k1_val=np.linspace(0.0,0.4,res1)
k2_val=np.linspace(0.0,1.5,res2)
k1,k2=np.meshgrid(k1_val,k2_val)
k1=k1.flatten()
k2=k2.flatten()

for i in range(res1*res2):
    temp1=np.array([k1[i],k2[i]])
    temp.append(iter(temp1))
#%% Parallel Calculations
result=[]
if __name__ =='__main__':
    pool=mp.Pool()
    result=pool.starmap(fig_3_static_grid_defs.RK4_static,(temp))
    pool.close()
    pool.join()
#%% Data saving
with open(r'Simulation_data/Figure_3/grid_data','wb') as fp:
    pickle.dump(result,fp)



