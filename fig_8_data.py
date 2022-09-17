#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.integrate import odeint, solve_ivp
import warnings
import matplotlib.pyplot as plt
from numba import njit
warnings.filterwarnings('ignore')

# %%

eps = 0.0001
alpha = np.pi*0.25
omega = 0.1
timeunits=200000
stepsize=0.01


# In[3]:


def gamma(k1, k2):
    return np.arctan2(np.sin(alpha)*(k1-k2), np.cos(alpha)*(k1+k2))


def A(k1, k2):
    return np.sqrt(np.power(k1, 2)+np.power(k2, 2)+2*k1*k2*np.cos(2*alpha))


def h1(theta, k1, k2):
    return a*np.sin(theta)


def h2(theta, k1, k2):
    return b*np.sin(theta-np.pi/2)


def system(y, t):  # Defintion of whole system
    [theta, k1, k2] = y
    return [omega-k1*np.sin(theta+alpha)-k2*np.sin(theta-alpha),
            -eps*(-h1(theta, k1, k2)+k1),
            -eps*(-h2(-theta, k1, k2)+k2)]


@njit
def gamma_jit(k):
    return np.arctan2(np.sin(alpha)*(k[0]-k[1]), np.cos(alpha)*(k[0]+k[1]))


@njit
def A_jit(k):
    return np.sqrt(np.power(k[0], 2)+np.power(k[1], 2)+2*k[0]*k[1]*np.cos(2*alpha))

@njit
def whole_flow(k,a,b):
    if A_jit(k)>omega:
        theta = np.arcsin(omega/A_jit(k))-gamma_jit(k)
        flow=np.array([a*np.sin(theta)-k[0],-b*np.cos(theta)-k[1],])
    else:
        c1 = np.cos(alpha)*(k[0]+k[1])
        c2 = np.sin(alpha)*(k[0]-k[1])
        c = np.power(c1, 2)+np.power(c2, 2)
        flow=np.array([a*c1*(omega-np.sqrt(omega**2-c))/c-k[0], -b*c2*(omega-np.sqrt(omega**2-c))/c-k[1]])
    return flow

@njit
def red_traj(k0,a,b, tmax, deltaT):
    tmax = tmax*eps
    t = np.linspace(0, tmax, int(tmax/np.abs(deltaT)))
    k = np.empty((2, len(t)))
    k[:, 0] = k0
    for i in range(len(t)-1):
        temp1 = whole_flow(k[:, i],a,b)
        temp2 = whole_flow(k[:, i]+deltaT*0.5*temp1,a,b)
        temp3 = whole_flow(k[:, i]+deltaT*0.5*temp2,a,b)
        temp4 = whole_flow(k[:, i]+deltaT*temp3,a,b)
        k[:, i+1] = k[:, i]+deltaT/6*(temp1+2*temp2+2*temp3+temp4)
    return k                                      


# %%
a = 0.5
b = 0.07

t = np.linspace(0, timeunits, int(timeunits/stepsize)+1)
y_LC_upper = odeint(system, [3.1339701353210216, -0.076457211, 0.063641222], t)
np.save(r'Simulation_data/Figure_4/y_LC_upper', y_LC_upper)


# %%
y1_red_upper = red_traj(np.array([0.15, 0.15]),a, b, timeunits, stepsize)
y2_red_upper = red_traj([-0.005, -0.005],a, b, timeunits, stepsize)
y_red_LC_upper = red_traj(y1_red_upper[:, -1],a, b, timeunits, stepsize)
plt.plot(y_red_LC_upper[0,:],y_red_LC_upper[1,:])

np.save(r'Simulation_data/Figure_4/y1_red_upper', y1_red_upper)
np.save(r'Simulation_data/Figure_4/y2_red_upper', y2_red_upper)
np.save(r'Simulation_data/Figure_4/y_red_LC_upper', y_red_LC_upper)

#%%
a = 0.385
b = 0.125

y_LC_lower = odeint(system, [3.20354479218193, -0.070347352, 0.0714165165], t)
np.save(r'Simulation_data/Figure_4/y_LC_lower', y_LC_lower)

y_00_lower = odeint(system, [0, 0.01, 0.01], t)
np.save(r'Simulation_data/Figure_4/y_00_lower', y_00_lower)

y1_red_lower = red_traj([0.15, 0.15],a, b, timeunits, stepsize)
y2_red_lower = red_traj([0.05, 0.025],a, b, timeunits, stepsize)
y3_red_lower = red_traj([-0.05, -0.025],a, b, timeunits, stepsize)
y00_red_lower = red_traj([0.01, 0.01],a, b, timeunits, stepsize)
y_unstable_temp = red_traj([0.01, 0.01],a, b, timeunits, -stepsize)
y_unstable_LC = red_traj([y_unstable_temp[0,-1], y_unstable_temp[1,-1]],a, b, timeunits, -stepsize)
y_red_LC_lower = red_traj(y1_red_lower[:, -1],a, b, timeunits, stepsize)
np.save(r'Simulation_data/Figure_4/y1_red_lower', y1_red_lower)
np.save(r'Simulation_data/Figure_4/y2_red_lower', y2_red_lower)
np.save(r'Simulation_data/Figure_4/y3_red_lower', y3_red_lower)
np.save(r'Simulation_data/Figure_4/y_unstable_LC', y_unstable_LC)
np.save(r'Simulation_data/Figure_4/y_red_LC_lower', y_red_LC_lower)
np.save(r'Simulation_data/Figure_4/y00_red_lower', y00_red_lower)
np.save(r'Simulation_data/Figure_4/t', t)
