# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:51:54 2021

@author: Max
"""

import numpy as np
from numba import njit

#%% Parameter for HHN
I1=np.array(5)
I2=np.array(13)
E_Na=np.array(50)
E_K=np.array(-77)
E_L=np.array(-54.4)
C=np.array(1)
g_Na=np.array(120)
g_K=np.array(36)
g_L=np.array(0.3)

E_r1=20
E_r2=20
k10=0.15
k20=1.0


#%% Parameter for Simulation

tmax=300000
chunksize=1000
chunk=1
deltaT=0.01
static=0




#%% Adaptationfunctions

A1=1.17
A2=0.4
tau1=0.25
tau2=1.1    

cp=1.5
cd=0.53
tau_p=1.8
tau_d=5

gamma=1

@njit
def STDP_plus_delayed(t,tau1):# hebbian STDP für postive deltaT
    return A1*np.exp(-t/tau1)*t**10*np.exp(10)/10**10/tau1**10

@njit
def STDP_minus_delayed(t):# hebbian STDP für negative deltaT
    return-A2*np.exp(t/tau2)*t**10*np.exp(10)/10**10/tau2**10

@njit
def k1_dot_plus(t,tau1):
    return -STDP_plus_delayed(t,tau1)

@njit
def k1_dot_minus(t):
    return -STDP_minus_delayed(t)

@njit
def test_func(t,gamma): #gamma denotes the amplitude of the adaptationfunction
    return gamma*((cp)*np.exp(-np.abs(t)/(tau_p))-cd*np.exp(-np.abs(t)/tau_d)+1/30)

@njit
def k2_dot_plus(t,gamma):
    return test_func(t,gamma)
@njit
def k2_dot_minus(t,gamma):
    return test_func(t,gamma)

#%%Simulation
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

#Initial conditions
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


@njit
def HHN(V1,m1,h1,n1,s1,V2,m2,h2,n2,s2,k1,k2):

    return np.array([(I1-g_Na*m1**3*h1*(V1-E_Na)-g_K*n1**4*(V1-E_K)-g_L*(V1-E_L)-0.5*(V1-E_r1)*k1*s2)/C,
    alpha_m(V1)*(1-m1)-beta_m(V1)*m1,
    alpha_h(V1)*(1-h1)-beta_h(V1)*h1,
    alpha_n(V1)*(1-n1)-beta_n(V1)*n1,
    (5-5*s1)/(1+np.exp(-0.125*V1+3*0.125))-s1,
    (I2-g_Na*m2**3*h2*(V2-E_Na)-g_K*n2**4*(V2-E_K)-g_L*(V2-E_L)-0.5*(V2-E_r2)*k2*s1)/C,
    alpha_m(V2)*(1-m2)-beta_m(V2)*m2,
    alpha_h(V2)*(1-h2)-beta_h(V2)*h2,
    alpha_n(V2)*(1-n2)-beta_n(V2)*n2,
    (5-5*s2)/(1+np.exp(-0.125*V2+3*0.125))-s2])

@njit
def RK4(gamma_var,tau1_var):
    gamma=gamma_var
    tau1=tau1_var
    delta=0.005
    if static==1:
        delta=0
    t=np.linspace(0,tmax,int(tmax/deltaT))
    y0=np.array([V01,m01,h01,n01,s01,V02,m02,h02,n02,s02])
    y=np.empty((len(y0),len(t)))
    k1=k10
    k2=k20
    k1_arr=np.array([k10])
    k2_arr=np.array([k20])
    k1_arr[0]=k10
    k2_arr[0]=k20
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
            temp1=HHN(y[0,i], y[1,i], y[2,i], y[3,i], y[4,i], y[5,i], y[6,i], y[7,i], y[8,i], y[9,i],k1_arr[len(k1_arr)-1],k2_arr[len(k2_arr)-1])
            temp2=HHN(y[0,i]+deltaT*0.5*temp1[0], y[1,i]+deltaT*0.5*temp1[1], y[2,i]+deltaT*0.5*temp1[2], y[3,i]+deltaT*0.5*temp1[3]
                           , y[4,i]+deltaT*0.5*temp1[4], y[5,i]+deltaT*0.5*temp1[5], y[6,i]+deltaT*0.5*temp1[6],
                           y[7,i]+deltaT*0.5*temp1[7], y[8,i]+deltaT*0.5*temp1[8], y[9,i]+deltaT*0.5*temp1[9],k1_arr[len(k1_arr)-1],k2_arr[len(k2_arr)-1])
            temp3=HHN(y[0,i]+deltaT*0.5*temp2[0], y[1,i]+deltaT*0.5*temp2[1], y[2,i]+deltaT*0.5*temp2[2], y[3,i]+deltaT*0.5*temp2[3]
                           , y[4,i]+deltaT*0.5*temp2[4], y[5,i]+deltaT*0.5*temp2[5], y[6,i]+deltaT*0.5*temp2[6],
                           y[7,i]+deltaT*0.5*temp2[7], y[8,i]+deltaT*0.5*temp2[8], y[9,i]+deltaT*0.5*temp2[9],k1_arr[len(k1_arr)-1],k2_arr[len(k2_arr)-1])
            temp4=HHN(y[0,i]+deltaT*temp3[0], y[1,i]+deltaT*temp3[1], y[2,i]+deltaT*temp3[2], y[3,i]+deltaT*temp3[3]
                           , y[4,i]+deltaT*temp3[4], y[5,i]+deltaT*temp3[5], y[6,i]+deltaT*temp3[6],
                           y[7,i]+deltaT*temp3[7], y[8,i]+deltaT*temp3[8], y[9,i]+deltaT*temp3[9],k1_arr[len(k1_arr)-1],k2_arr[len(k2_arr)-1])
            y[:,i+1]=y[:,i]+deltaT*(1/6*temp1+1/3*temp2+1/3*temp3+1/6*temp4)
            if y[0,i]-y[0,i-1] >0 and y[0,i+1]-y[0,i]<0 and y[0,i]>0 and i!=0:
                N1_spikes=np.append(N1_spikes,t[i])
                k1=k1+delta*k1_dot_plus(t[i]-N2_spikes[len(N2_spikes)-1],tau1)
                k2=k2+delta*k2_dot_minus(-(t[i]-N2_spikes[len(N2_spikes)-1]),gamma) 
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
                t_update=np.append(t_update,t[i])
                j=j+1
            if y[5,i]-y[5,i-1] >0 and y[5,i+1]-y[5,i]<0 and y[5,i]>0 and i!=0:
                N2_spikes=np.append(N2_spikes,t[i])
                k1=k1+delta*k1_dot_minus(-(t[i]-N1_spikes[len(N1_spikes)-1]))
                k2=k2+delta*k2_dot_plus((t[i]-N1_spikes[len(N1_spikes)-1]),gamma)
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
                t_update=np.append(t_update,t[i])
                j=j+1
                if N1_spikes[len(N1_spikes)-1]==N2_spikes[len(N2_spikes)-1] and len(N1_spikes)>3:
                    k1_arr=k1_arr[:len(k1_arr)-2]
                    k2_arr=k2_arr[:len(k2_arr)-2]
                    t_update=t_update[:len(t_update)-2]
        y0=y[:,i+1]
    return N1_spikes,N2_spikes


        
@njit
def order_parameter(N1_spikes,N2_spikes):
    max_t=np.min(np.array([N1_spikes[-1],N2_spikes[-1]]))
    print (max_t)
    t=np.linspace(150000,max_t,int((max_t-150000)/deltaT)+1)
    delta_T=np.empty(2)
    phases=np.empty(2)
    ord=np.empty(len(t))
    k=np.empty(2)
    k[:]=int(0)
    for j in range(len(t)):
        if t[j]>N1_spikes[int(k[0]+1)]:
            k[0]=int(k[0]+1)
        elif t[j]>N2_spikes[int(k[1]+1)]:
            k[1]=int(k[1]+1)
        delta_T[0]=t[j]-N1_spikes[int(k[0])]
        delta_T[1]=t[j]-N2_spikes[int(k[1])]
        if N1_spikes[int(k[0]+1)]!=N1_spikes[int(k[0])]:
            phases[0]=delta_T[0]/(N1_spikes[int(k[0]+1)]-N1_spikes[int(k[0])])*2*np.pi
        if N2_spikes[int(k[0]+1)]!=N2_spikes[int(k[0])]:
            phases[1]=delta_T[1]/(N2_spikes[int(k[1]+1)]-N2_spikes[int(k[1])])*2*np.pi
        ord[j]=1/2*np.abs(np.sum(np.exp(phases*1j)))
    return ord,max_t



@njit
def burst_finder(gamma_var,tau1_var):
    N1_spikes,N2_spikes=RK4(gamma_var,tau1_var)
    order,max_t=order_parameter(N1_spikes[N1_spikes>150000],N2_spikes[N2_spikes>150000])
    if np.all((order<np.mean(order)+0.1) & (order>np.mean(order)-0.1)):
        bla=2
    else:
        eq=np.empty(int((max_t-150000)/1000))
        for i in range(len(eq)):
            temp=order[int(i*1000/deltaT):int((i+1)*1000/deltaT)]
            if np.all((temp<np.mean(temp)+0.1) & (temp>np.mean(temp)-0.1)):
                eq[i]=0
            else:
                eq[i]=1
        synch=np.where(eq==0)[0]
        asynch=np.where(eq==1)[0]
        if len(synch)<5:
            bla=0
        elif len(asynch)<5:
            bla=2
        elif len(synch)>5 and len(asynch)>5 and np.max(asynch)>np.min(synch) and np.max(synch)>np.min(asynch):
            bla=1
        else:
            bla=3
    return gamma_var,tau1_var,bla,synch,asynch
        


