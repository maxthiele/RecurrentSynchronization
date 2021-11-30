# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:47:49 2021

@author: Max
"""

#%%
import numpy as np
from numba import njit

#%% Parameter node dynamics

I1=np.array(5)
I2=np.array(13)
I=np.append(I1,I2)
E_Na=np.array(50)
E_K=np.array(-77)
E_L=np.array(-54.4)
C=np.array(1)
g_Na=np.array(120)
g_K=np.array(36)
g_L=np.array(0.3)

E_r=20

#%% Simulation parameters

tmax=50000
deltaT=0.01
chunksize=1000

#%% Definitions

@njit
def alpha_m(V):
    return (0.1*V+4)/(1-np.exp(-0.1*V-4))

@njit
def beta_m(V):
    return 4*np.exp(-V/18-65/18)

@njit    
def alpha_h(V):
    return 0.07*np.exp(-0.05*V-65/20)

@njit
def beta_h(V):
    return 1/(1+np.exp(-0.2*V-3.5))

@njit
def alpha_n(V):
    return (0.01*V+0.55)/(1-np.exp(-0.1*V-5.5))

@njit
def beta_n(V):
    return 0.125*np.exp(-0.0125*V-65/80)


@njit
def dydt(y,k):
    dy=np.empty(len(y))
    for i in range(2):
        dy[i*5]=I[i]-g_Na*y[i*5+1]**3*y[i*5+2]*(y[i*5]-E_Na)-g_K*y[i*5+3]**4*(y[i*5]-E_K)-g_L*(y[i*5]-E_L)-(y[i*5]-E_r)/2*np.sum(k[i]*y[4::5])      
        dy[i*5+1]=alpha_m(y[i*5])*(1-y[i*5+1])-beta_m(y[i*5])*y[i*5+1]
        dy[i*5+2]=alpha_h(y[i*5])*(1-y[i*5+2])-beta_h(y[i*5])*y[i*5+2]
        dy[i*5+3]=alpha_n(y[i*5])*(1-y[i*5+3])-beta_n(y[i*5])*y[i*5+3]
        dy[i*5+4]=5*(1-y[i*5+4])/(1+np.exp((-y[i*5]+3)/8))-y[i*5+4]
    return dy

@njit
def RK4_static(k10,k20):
    
    V01=-70
    m01=alpha_m(V01)/(alpha_m(V01)+beta_m(V01))
    h01=alpha_h(V01)/(alpha_h(V01)+beta_h(V01))
    n01=alpha_n(V01)/(alpha_n(V01)+beta_n(V01))
    s01=0.0
    V02=-70
    m02=alpha_m(V02)/(alpha_m(V02)+beta_m(V02))
    h02=alpha_h(V02)/(alpha_h(V02)+beta_h(V02))
    n02=alpha_n(V02)/(alpha_n(V02)+beta_n(V02))
    s02=0.0
    y0=np.array([V01,m01,h01,n01,s01,V02,m02,h02,n02,s02])
    k=np.array([k10,k20])
    j=1
    N1_spikes=np.array([0.0])
    N2_spikes=np.array([0.0])
    chunks=int(tmax/chunksize)
    for j in range(chunks):
        t_start=chunksize*j
        t_end=chunksize*(j+1)
        t=np.linspace(t_start,t_end,int(chunksize/deltaT))
        y=np.empty((len(y0),len(t)))
        y[:,0]=y0
        for i in range(len(t)):
            temp1=dydt(y[:,i],k)
            temp2=dydt(y[:,i]+deltaT*0.5*temp1,k)
            temp3=dydt(y[:,i]+deltaT*0.5*temp2,k)
            temp4=dydt(y[:,i]+deltaT*temp3,k)
                    
            y[:,i+1]=y[:,i]+deltaT/6*(temp1+2*temp2+2*temp3+temp4)
            
            
            if y[0,i]-y[0,i-1] >0 and y[0,i+1]-y[0,i]<0 and y[0,i]>0 and i!=0:
                N1_spikes=np.append(N1_spikes,t[i])
            if y[5,i]-y[5,i-1] >0 and y[5,i+1]-y[5,i]<0 and y[5,i]>0 and i!=0:
                N2_spikes=np.append(N2_spikes,t[i])

        y0=y[:,i+1]   
    N1_spikes_chunk=N1_spikes[N1_spikes >tmax-20000+1]
    N2_spikes_chunk=N2_spikes[N2_spikes >tmax-20000+1]
    ISI_1_aver=np.mean(N1_spikes_chunk[1:]-N1_spikes_chunk[0:len(N1_spikes_chunk)-1]) # Inter-Spike-Intervalls
    ISI_2_aver=np.mean(N2_spikes_chunk[1:]-N2_spikes_chunk[0:len(N2_spikes_chunk)-1])
    if np.abs(ISI_1_aver-ISI_2_aver)<0.002: #phase locked solution
        GG=2
    elif np.abs(ISI_1_aver-ISI_2_aver)>0.02: # asynchronous solution
        GG=0
    else: # something in between
        GG=1

    return GG,k10,k20,N1_spikes_chunk,N2_spikes_chunk
    

