# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 11:13:39 2021

@author: Max
"""

import numpy as np
from numba import njit
import pickle
import multiprocessing as mp


#%% Loading spiking times of static grid

with open(r'Simulation_data/Figure_3/grid_data','rb') as fp:
    result=pickle.load(fp)
    
k1_res=5
k2_res=5

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



#%% Parameter Adaptation functions

A1=1.17
A2=0.4
tau1=0.25
tau2=1.1    

cp=1.5
cd=0.53
tau_p=1.8
tau_d=5

#%% Definitions Simulation

@njit
def STDP_plus_delayed(t):# hebbian STDP für postive deltaT
    return A1*np.exp(-t/tau1)*t**10*np.exp(10)/10**10/tau1**10

@njit
def STDP_minus_delayed(t):# hebbian STDP für negative deltaT
    return-A2*np.exp(t/tau2)*t**10*np.exp(10)/10**10/tau2**10
    
    
@njit
def k1_dot_plus(t):
    return -STDP_plus_delayed(t)

@njit
def k1_dot_minus(t):
    return -STDP_minus_delayed(t)

@njit
def mexican_hat(t):
    return ((cp)*np.exp(-np.abs(t)/(tau_p))-cd*np.exp(-np.abs(t)/tau_d)+1/30)

@njit
def k2_dot_plus(t):
    return mexican_hat(t)

@njit
def k2_dot_minus(t):
    return mexican_hat(t)

def flow(k1, k2, Delta_T12, Delta_T21, phaselag_dist,GG):
    k1_flow=np.empty(len(GG))
    k2_flow=np.empty(len(GG))
    for i in range(len(GG)):     
        k1_flow[i]=(np.mean(k1_dot_plus(-Delta_T21[i])*len(Delta_T21[i]))+np.mean(k1_dot_minus(-Delta_T12[i]))*len(Delta_T12[i]))/(len(Delta_T21[i])+len(Delta_T12[i]))
        k2_flow[i]=(np.mean(k2_dot_plus(Delta_T12[i])*len(Delta_T12[i]))+np.mean(k2_dot_minus(Delta_T21[i]))*len(Delta_T21[i]))/(len(Delta_T21[i])+len(Delta_T12[i]))

    return k1_flow,k2_flow


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
def RK4(y0,k10,k20,tmax,deltaT,chunk,chunksize,static):
    delta=0.005
    if static==1:
        delta=0
    t=np.linspace(0,tmax,int(tmax/deltaT))
    y=np.empty((len(y0),len(t)))
    k1=k10
    k2=k20
    k1_arr=np.array([k10])
    k2_arr=np.array([k20])
    k1_arr[0]=k10
    k2_arr[0]=k20
    k=np.array([k10,k20])
    j=1
    y[:,0]=y0
    N1_spikes=np.array([0.0])
    N2_spikes=np.array([0.0])
    t_update=np.array([0.0])
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
                k1=k1+delta*k1_dot_plus(t[i]-N2_spikes[len(N2_spikes)-1])
                k2=k2+delta*k2_dot_minus(-(t[i]-N2_spikes[len(N2_spikes)-1])) 
                if k1>1.5:
                    k1=1.5
                elif k1<0.0:
                    k1=0.0
                if k2>1.5:
                    k2=1.5
                elif k2<0.0:
                    k2=0.0
                k1_arr=np.append(k1_arr,k1)
                k2_arr=np.append(k2_arr,k2)
                k=np.array([k1,k2])
                t_update=np.append(t_update,t[i])
                j=j+1
            if y[5,i]-y[5,i-1] >0 and y[5,i+1]-y[5,i]<0 and y[5,i]>0 and i!=0:
                N2_spikes=np.append(N2_spikes,t[i])
                k1=k1+delta*k1_dot_minus(-(t[i]-N1_spikes[len(N1_spikes)-1]))
                k2=k2+delta*k2_dot_plus((t[i]-N1_spikes[len(N1_spikes)-1]))
                if k1>1.5:
                    k1=1.5
                elif k1<0.0:
                    k1=0.0
                if k2>1.5:
                    k2=1.5
                elif k2<0.0:
                    k2=0.0
                k1_arr=np.append(k1_arr,k1)
                k2_arr=np.append(k2_arr,k2)
                k=np.array([k1,k2])
                t_update=np.append(t_update,t[i])
                j=j+1
                if N1_spikes[len(N1_spikes)-1]==N2_spikes[len(N2_spikes)-1] and len(N1_spikes)>3:
                    k1_arr=k1_arr[:len(k1_arr)-2]
                    k2_arr=k2_arr[:len(k2_arr)-2]
                    t_update=t_update[:len(t_update)-2]
        y0=y[:,i+1]
    return y,t,k1_arr,k2_arr,N1_spikes,N2_spikes,t_update




@njit
def order_parameter(N1_spikes,N2_spikes):
    t=np.linspace(0,tmax,int(tmax/delta_T)+1)
    deltaT=np.empty(2)
    phases=np.empty(2)
    ord=np.empty(len(t))
    k=np.empty(2)
    k[:]=int(0)
    for j in range(len(t)):
        if t[j]>N1_spikes[int(k[0]+1)]:
            k[0]=int(k[0]+1)
        elif t[j]>N2_spikes[int(k[1]+1)]:
            k[1]=int(k[1]+1)
        deltaT[0]=t[j]-N1_spikes[int(k[0])]
        deltaT[1]=t[j]-N2_spikes[int(k[1])]
        if N1_spikes[int(k[0]+1)]!=N1_spikes[int(k[0])]:
            phases[0]=deltaT[0]/(N1_spikes[int(k[0]+1)]-N1_spikes[int(k[0])])*2*np.pi
        if N2_spikes[int(k[0]+1)]!=N2_spikes[int(k[0])]:
            phases[1]=deltaT[1]/(N2_spikes[int(k[1]+1)]-N2_spikes[int(k[1])])*2*np.pi
        ord[j]=1/2*np.abs(np.sum(np.exp(phases*1j)))
    return ord

#%% Definitions: Analysis of grid data
@njit
def nearest_spike(a1,a2): #return the nearest spiking time for every spike of the two neurons.
    Delta_T12=np.empty(0)
    Delta_T21=np.empty(0)
    for i in np.arange(len(a1)-2)+1:
        val=a1[i]
        idx1 = (np.abs(a2-val)).argmin()
        if a2[idx1]>val:
            Delta_T12=np.append(Delta_T12,a2[idx1]-val)
            Delta_T21=np.append(Delta_T21,a2[idx1-1]-val)
            if a2[idx1+1]<a1[i+1]:
                Delta_T12=np.append(Delta_T12,a2[idx1+1]-val)
        elif a2[idx1]<val:
            Delta_T12=np.append(Delta_T12,a2[idx1+1]-val)
            Delta_T21=np.append(Delta_T21,a2[idx1]-val)
    return Delta_T12[1:-2], Delta_T21[1:-2]

def determine_phaselags(result):
    phaselag=np.empty(0)
    phaselag_dist=[]
    GG=np.empty(0)
    k1=np.empty(0)
    k2=np.empty(0)
    k1_dist=np.empty(0)
    k2_dist=np.empty(0)
    Delta_T12=[]
    Delta_T21=[]
    for i in range(len(result)):
        temp=result[i]
        GG=np.append(GG,temp[0])
        k1=np.append(k1,temp[1])
        k2=np.append(k2,temp[2])
        N1_spikes=np.array(temp[3])
        N2_spikes=np.array(temp[4])
        N1_spikes_last=N1_spikes[(N1_spikes>25000)]
        N2_spikes_last=N2_spikes[(N2_spikes>25000)]
        if temp[0]==2:
            temp1,temp2=nearest_spike(N1_spikes_last,N2_spikes_last)
            Delta_T12.append(temp1)
            Delta_T21.append(temp2)
            temp3=np.append(temp1,temp2)
            temp3=temp3.flatten()
            phaselag=np.append(phaselag,np.mean(-Delta_T21[i]))
            phaselag_dist.append(temp3)
            k1_dist=np.append(k1_dist,temp[1])
            k2_dist=np.append(k2_dist,temp[2])
        if temp[0]==0:
             temp1,temp2=nearest_spike(N1_spikes_last,N2_spikes_last)
             Delta_T12.append(temp1)
             Delta_T21.append(temp2)
             temp3=np.append(temp1,temp2)
             temp3=temp3.flatten()
             phaselag=np.append(phaselag,-3)
             phaselag_dist.append(temp3)
             k1_dist=np.append(k1_dist,temp[1])
             k2_dist=np.append(k2_dist,temp[2])
        if temp[0]==1:
            temp1,temp2=nearest_spike(N1_spikes_last,N2_spikes_last)
            Delta_T12.append(temp1)
            Delta_T21.append(temp2)
            phaselag=np.append(phaselag,np.mean(-Delta_T21[i]))
            k1_dist=np.append(k1_dist,temp[1])
            k2_dist=np.append(k2_dist,temp[2])
    return k1, k2, k1_dist, k2_dist, phaselag, phaselag_dist,GG, Delta_T12, Delta_T21

#%%
k1, k2, k1_dist, k2_dist, phaselag, phaselag_dist,GG, Delta_T12, Delta_T21=determine_phaselags(result)



#%% Initial conditions
V01=np.random.uniform(-70,20)
m01=alpha_m(V01)/(alpha_m(V01)+beta_m(V01))
h01=alpha_h(V01)/(alpha_h(V01)+beta_h(V01))
n01=alpha_n(V01)/(alpha_n(V01)+beta_n(V01))
s01=0.0

V02=np.random.uniform(-70,20)
m02=alpha_m(V02)/(alpha_m(V02)+beta_m(V02))
h02=alpha_h(V02)/(alpha_h(V02)+beta_h(V02))
n02=alpha_n(V02)/(alpha_n(V02)+beta_n(V02))
s02=0.0
y02=np.array([V01,m01,h01,n01,s01,V02,m02,h02,n02,s02])


k10=0.15
k20=1.0

#%% Calculations

tmax=200000
delta_T=0.01

sol2,t,k1_arr,k2_arr,N1_spikes,N2_spikes,t_update=RK4(y02,k10,k20,tmax,delta_T,1,1000,0)

k1_flow,k2_flow=flow(k1, k2, Delta_T12, Delta_T21, phaselag_dist,GG)

order=order_parameter(N1_spikes,N2_spikes)

#%% Saving of data

np.save(r'Simulation_data/Figure_3/sol2',sol2)
np.save(r'Simulation_data/Figure_3/t',t)
np.save(r'Simulation_data/Figure_3/k1_arr',k1_arr)
np.save(r'Simulation_data/Figure_3/k2_arr',k2_arr)
np.save(r'Simulation_data/Figure_3/N1_spikes',N1_spikes)
np.save(r'Simulation_data/Figure_3/N2_spikes',N2_spikes)
np.save(r'Simulation_data/Figure_3/t_update',t_update)
np.save(r'Simulation_data/Figure_3/k1',k1)
np.save(r'Simulation_data/Figure_3/k2',k2)
np.save(r'Simulation_data/Figure_3/k1_flow',k1_flow)
np.save(r'Simulation_data/Figure_3/k2_flow',k2_flow)
np.save(r'Simulation_data/Figure_3/GG',GG)
np.save(r'Simulation_data/Figure_3/order',order)













