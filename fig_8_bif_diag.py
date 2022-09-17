# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:15:44 2021

@author: Max
"""

# In[1]:


import numpy as np
from scipy.integrate import quad,odeint,solve_ivp
from scipy.optimize import fsolve,root
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[2]:


def gamma(k1,k2):
    return np.arctan2(np.sin(alpha)*(k1-k2),np.cos(alpha)*(k1+k2))
    
def A(k1,k2):
    return np.sqrt(np.power(k1,2)+np.power(k2,2)+2*k1*k2*np.cos(2*alpha))

def h1(theta,k1,k2): 
    return a*np.sin(theta)
    
def h2(theta,k1,k2):
    return b*np.sin(theta-np.pi/2)

def system(y,t):
    [theta,k1,k2]=y
    return [omega-k1*np.sin(theta+alpha)-k2*np.sin(theta-alpha),
            -eps*(-h1(theta,k1,k2)+k1),
            -eps*(-h2(-theta,k1,k2)+k2)]


#

# In[4]:


def cond(y):
    return np.any((A(y[n_cutoff:n-2,1],y[n_cutoff:n-2,2])>omega) & (np.roll(A(y[n_cutoff:n-2,1],y[n_cutoff:n-2,2]),-1)<omega))
        
eps=0.0001
alpha=np.pi*0.25
omega=0.10
a2=0
b1=0
[a_0,b_0]=np.array([-0.15,1])
stepsize=0.01
timeunits=200000
n=int(timeunits/stepsize)
t=np.linspace(0,timeunits,n)
cutoff=timeunits-100000
n_cutoff=int(cutoff/stepsize)


# In[ ]:


direction=np.array([0,1])
a=0.4
b=0.15
step=0.01
a_all=np.empty(0)
b_all=np.empty(0)
IC=np.array([0,-0.5,0.5])
y=odeint(system,IC,t)
cond_bool=1
psi=-np.pi/2
dir_bool=0
dreh=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])


while cond_bool:
    a=a+step*direction[0]
    b=b+step*direction[1]
    print(a,b)
    IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
    y=odeint(system,IC,t)
    cond_bool=cond(y)
    if cond_bool==1:
        IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
    else:
        break
direction=np.array([0,1])
direction_all=np.empty(0)
start_boundary=np.array([a,b])
i=0
while np.abs(a-start_boundary[0])>0.01 or np.abs(b-start_boundary[1])>0.01 or i<50:
    if i%10==0:
        print(i,np.around(a,decimals=2),np.around(b,decimals=2))
    if i>200:
        break
    i=i+1
    direction_all=np.append(direction_all,direction)
    direction1=np.dot(dreh,direction)
    a=np.around(a+step*direction1[0],decimals=2)
    b=np.around(b+step*direction1[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            direction=direction1
            dir_bool=0
            continue
    a=np.around(a+step*direction[0]-step*direction1[0],decimals=2)
    b=np.around(b+step*direction[1]-step*direction1[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            continue
    direction2=np.dot(-dreh,direction)
    a=np.around(a+step*direction2[0],decimals=2)
    b=np.around(b+step*direction2[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            direction=direction2
            dir_bool=0
            continue
    if dir_bool==0:
        direction=-np.array(direction)
        dir_bool=1

        
#%%
np.save(r'Simulation_data/Figure_4/bursting_data_a_b_grid/a_burst1',a_all)
np.save(r'Simulation_data/Figure_4/bursting_data_a_b_grid/b_burst1',b_all)
#%%
direction=np.array([1,0])
a=0.15
b=0.4
step=0.01
a_all=np.empty(0)
b_all=np.empty(0)
IC=np.array([0,-0.5,0.5])
y=odeint(system,IC,t)
cond_bool=1
psi=np.pi/2
dir_bool=0
dreh=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])


while cond_bool:
    a=a+step*direction[0]
    b=b+step*direction[1]
    print(a,b)
    IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
    y=odeint(system,IC,t)
    cond_bool=cond(y)
    if cond_bool==1:
        IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
    else:
        break
direction=np.array([0,1])
direction_all=np.empty(0)
start_boundary=np.array([a,b])
i=0
while np.abs(a-start_boundary[0])>0.01 or np.abs(b-start_boundary[1])>0.01 or i<50:
    if i%10==0:
        print(i,np.around(a,decimals=2),np.around(b,decimals=2))
    if i>200:
        break
    i=i+1
    direction_all=np.append(direction_all,direction)
    direction1=np.dot(dreh,direction)
    a=np.around(a+step*direction1[0],decimals=2)
    b=np.around(b+step*direction1[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            direction=direction1
            dir_bool=0
            continue
    a=np.around(a+step*direction[0]-step*direction1[0],decimals=2)
    b=np.around(b+step*direction[1]-step*direction1[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            continue
    direction2=np.dot(-dreh,direction)
    a=np.around(a+step*direction2[0],decimals=2)
    b=np.around(b+step*direction2[1],decimals=2)
    if a<0.5+step/2 and a>-0.5-step/2 and b<0.5+step/2 and b>-0.5-step/2:
        y=odeint(system,IC,t)
        if cond(y):
            a_all=np.append(a_all,a)
            b_all=np.append(b_all,b)
            IC=np.array([y[n-1,0]%(2*np.pi),y[n-1,1],y[n-1,2]])
            direction=direction2
            dir_bool=0
            continue
    if dir_bool==0:
        direction=-np.array(direction)
        dir_bool=1

        
#%%
np.save(r'Simulation_data/Figure_4/bursting_data_a_b_grid/a_burst2',a_all)
np.save(r'Simulation_data/Figure_4/bursting_data_a_b_grid/b_burst2',b_all)


