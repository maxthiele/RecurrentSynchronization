# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:04:40 2021

@author: Max
"""

import numpy as np
from numba import njit
from scipy.ndimage import uniform_filter1d
import sys

#%% Parameter node dynamics

I1=np.array(5)
I2=np.array(13)
E_Na=np.array(50)
E_K=np.array(-77)
E_L=np.array(-54.4)
C=np.array(1)
g_Na=np.array(120)
g_K=np.array(36)
g_L=np.array(0.3)

E_r=20
n1=100
n2=100
N=n1+n2

I=np.append(np.ones(n1)*I1,np.ones(n2)*I2)+np.random.uniform(-0.01,0.01,N)


#%% Parameter Adaptation functions
A1=1.17
A2=0.4
tau1=0.25
tau2=1.1    

cp=1.5
cd=0.53
tau_p=1.8
tau_d=5



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
def plas_func1(t):
    if t==0:
        val=0
    elif t>0: 
        val=-A1*np.exp(-t/tau1)*t**10*np.exp(10)/10**10/tau1**10
    else: 
        val=-(-A2*np.exp(t/tau2)*t**10*np.exp(10)/10**10/tau2**10)
    return val

@njit
def plas_func2(t):
    if t==0:
        val=0
    elif t>0: 
        val=((cp)*np.exp(-np.abs(t)/(tau_p))-cd*np.exp(-np.abs(t)/tau_d)+1/30)
    else: 
        val=((cp)*np.exp(-np.abs(t)/(tau_p))-cd*np.exp(-np.abs(t)/tau_d)+1/30)
    return val

@njit
def dydt(y,k):
    dy=np.empty(len(y))
    for i in range(N):
        dy[i*5]=I[i]-g_Na*y[i*5+1]**3*y[i*5+2]*(y[i*5]-E_Na)-g_K*y[i*5+3]**4*(y[i*5]-E_K)-g_L*(y[i*5]-E_L)-(y[i*5]-E_r)/N*np.sum(k[i,:]*y[4::5])      
        dy[i*5+1]=alpha_m(y[i*5])*(1-y[i*5+1])-beta_m(y[i*5])*y[i*5+1]
        dy[i*5+2]=alpha_h(y[i*5])*(1-y[i*5+2])-beta_h(y[i*5])*y[i*5+2]
        dy[i*5+3]=alpha_n(y[i*5])*(1-y[i*5+3])-beta_n(y[i*5])*y[i*5+3]
        dy[i*5+4]=5*(1-y[i*5+4])/(1+np.exp((-y[i*5]+3)/8))-y[i*5+4]
    
    return dy   
    
@njit      
def RK4(y0,k0,tmax,deltaT,chunksize):
    k=np.copy(k0)
    delta=0.005
    N_spikes=np.zeros(N)
    spiketimes=np.empty((N,int(tmax/10)))
    spiketimes[:,:]=np.nan
    chunks=int(tmax/chunksize)
    for ii in range(chunks):
        t_start=chunksize*ii
        t_end=chunksize*(ii+1)
        t=np.linspace(t_start,t_end,int(chunksize/deltaT)+1)
        y=np.empty((len(y0),len(t)))
        y[:,0]=y0
        for i in range(len(t)-1):
            temp1=dydt(y[:,i],k)
            temp2=dydt(y[:,i]+deltaT*0.5*temp1,k)
            temp3=dydt(y[:,i]+deltaT*0.5*temp2,k)
            temp4=dydt(y[:,i]+deltaT*temp3,k)
                    
            y[:,i+1]=y[:,i]+deltaT/6*(temp1+2*temp2+2*temp3+temp4)
    
            spike,=np.where(((y[0::5,i+1]>0)) & (y[0::5,i]<0)) 
            if spike.size!=0:
                N_spikes[spike]=t[i]
                for j in spike:
                    for l in range(N):
                        if j<n1:
                            k[j,l]=k[j,l]+delta*plas_func1(t[i]-N_spikes[l])
                            if l<n1+1:
                                k[l,j]=k[l,j]+delta*plas_func1(-(t[i]-N_spikes[l]))
                            else:
                                k[l,j]=k[l,j]+delta*plas_func2(-(t[i]-N_spikes[l]))
                        else:
                            k[j,l]=k[j,l]+delta*plas_func2(t[i]-N_spikes[l])
                            if l<n1+1:
                                k[l,j]=k[l,j]+delta*plas_func1(-(t[i]-N_spikes[l]))
                            else:
                                k[l,j]=k[l,j]+delta*plas_func2(-(t[i]-N_spikes[l]))
                    k[j,:][k[j,:]>1.5]=1.5 
                    k[:,j][k[:,j]>1.5]=1.5
                    k[j,:][k[j,:]<0]=0 
                    k[:,j][k[:,j]<0]=0
                    temp=spiketimes[j,:]
                    temp=temp[~np.isnan(temp)]
                    temp=np.append(temp,t[i])
                    spiketimes[j,:len(temp)]=temp
            y0=y[:,i+1]
    return k,spiketimes

@njit
def order_parameter(spiketimes,t):
    deltaT=np.empty(N)
    phases=np.empty(N)
    ord=np.empty(len(t))
    k=np.empty(N)
    k[:]=int(0)
    for j in range(len(t)):
        for i in range(N):
            if t[j]>spiketimes[i,int(k[i]+1)]:
                k[i]=int(k[i]+1)
            deltaT[i]=t[j]-spiketimes[i,int(k[i])]
            phases[i]=deltaT[i]/(spiketimes[i,int(k[i]+1)]-spiketimes[i,int(k[i])])*2*np.pi
        ord[j]=1/N*np.abs(np.sum(np.exp(phases*1j)))
    return ord

@njit
def spike_rate_2_pop(spiketimes,interval):
    spikerate=np.empty(int(tmax/interval))
    spikerate1=np.empty(int(tmax/interval))
    spikerate2=np.empty(int(tmax/interval))
    spiketimes1=spiketimes[:n1,:]
    spiketimes2=spiketimes[n1:,:]
    spiketimes1=spiketimes1.flatten()
    spiketimes2=spiketimes2.flatten()
    t_int=np.arange(0,tmax,interval)+interval/2
    
    for i in np.arange(int(tmax/interval)):
        spikerate1[i]=len(spiketimes1[((spiketimes1>i*interval-0.04) & (spiketimes1<(i+1)*interval))])/n1
        spikerate2[i]=len(spiketimes2[((spiketimes2>i*interval-0.04) & (spiketimes2<(i+1)*interval))])/n2
        spikerate[i]=(n1*spikerate1[i]+n2*spikerate2[i])/N
    return spikerate,spikerate1,spikerate2,t_int

#%% Initial conditions

if sys.argv[1]=='random':    
    y0=np.empty(0)
    for i  in range(N):
        V01=np.random.uniform(-70,20)
        m01=alpha_m(V01)/(alpha_m(V01)+beta_m(V01))
        h01=alpha_h(V01)/(alpha_h(V01)+beta_h(V01))
        n01=alpha_n(V01)/(alpha_n(V01)+beta_n(V01))
        s01=0.0
        y0=np.append(y0,np.array([V01,m01,h01,n01,s01,]))
    
    k0=np.zeros((N,N))+np.random.uniform(0.0,0.5,(N,N))
    k0=k0-np.diag(np.diag(k0))
    
elif sys.argv[1]=='chosen':
    k0=np.load(r'fig_2_IC_k.npy')
    y0=np.load(r'fig_2_IC_y.npy')

#%% Simulation of the system and calculation of the order parameter and spikerates

tmax=3000
t=np.linspace(0,tmax,int(tmax/0.04)+1)
k,spiketimes=RK4(y0,k0,tmax,0.04,1000)
order=order_parameter(spiketimes,t)
spikerate,spikerate1,spikerate2, t_int=spike_rate_2_pop(spiketimes,3000)


#%% Saving Data

np.save(r'Simulation_data/Figure_2/k',k)
np.save(r'Simulation_data/Figure_2/spiketimes',spiketimes)
np.save(r'Simulation_data/Figure_2/order',order)
np.save(r'Simulation_data/Figure_2/t',t)
np.save(r'Simulation_data/Figure_2/spikerate',spikerate)
np.save(r'Simulation_data/Figure_2/spikerate_pop_1',spikerate1)
np.save(r'Simulation_data/Figure_2/spikerate_pop_2',spikerate2)
np.save(r'Simulation_data/Figure_2/t_int',t_int)




