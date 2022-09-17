# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:04:09 2021

@author: Max
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:44:57 2021

@author: Max
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import pickle
import matplotlib.cm as cm
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%

 
k1_res=5
k2_res=5


with open(r'Simulation_data/Figure_3/bif_diagram_data','rb') as fp:
    bif_data=pickle.load(fp)

GG_bif=np.empty(0)
gamma_bif=np.empty(0)
tau_bif=np.empty(0)
for i in range(len(bif_data)):
    temp=bif_data[i]
    GG_bif=np.append(GG_bif,temp[2])
    gamma_bif=np.append(gamma_bif,temp[0])
    tau_bif=np.append(tau_bif,temp[1])
gamma_bif=np.reshape(gamma_bif,(10,10))
tau_bif=np.reshape(tau_bif,(10,10))
GG_bif=np.reshape(GG_bif,(10,10))
#%%
sol2=np.load(r'Simulation_data/Figure_3/sol2.npy')
k1_arr=np.load(r'Simulation_data/Figure_3/k1_arr.npy')
k2_arr=np.load(r'Simulation_data/Figure_3/k2_arr.npy')
N1_spikes=np.load(r'Simulation_data/Figure_3/N1_spikes.npy')
N2_spikes=np.load(r'Simulation_data/Figure_3/N2_spikes.npy')
t_update=np.load(r'Simulation_data/Figure_3/t_update.npy')
k1_flow=np.load(r'Simulation_data/Figure_3/k1_flow.npy')
k2_flow=np.load(r'Simulation_data/Figure_3/k2_flow.npy')
k1=np.load(r'Simulation_data/Figure_3/k1.npy')
k2=np.load(r'Simulation_data/Figure_3/k2.npy')
GG=np.load(r'Simulation_data/Figure_3/GG.npy')
order=np.load(r'Simulation_data/Figure_3/order.npy')



#%%
def fig_2HHN_plot(k1,k2,k1_flow,k2_flow, GG, k1_arr, k2_arr,N1_spikes,N2_spikes,t_update,t_anf,t_end):
    t=t=np.linspace(0,200000,int(200000/0.01)+1)
    ord_par=order[(t>=t_anf) & (t<=t_end)]
    N1_spikes=N1_spikes[(N1_spikes>t_anf) & (N1_spikes<t_end)]
    N2_spikes=N2_spikes[(N2_spikes>t_anf) & (N2_spikes<t_end)]
    k1_arr=k1_arr[(t_update>t_anf) & (t_update<t_end)]
    k2_arr=k2_arr[(t_update>t_anf) & (t_update<t_end)]
    t_update=t_update[(t_update>t_anf) & (t_update<t_end)]/1000
    t=t[(t>=t_anf) & (t<=t_end)]/1000
    
    ind=np.lexsort((k1,k2))
    k1_grid=k1[ind]
    k2_grid=k2[ind]
    
    k1_flow=k1_flow[ind]
    k2_flow=k2_flow[ind]
    GG_temp=GG[ind]
    k1_grid=np.reshape(k1_grid,(k1_res,k2_res))
    k2_grid=np.reshape(k2_grid,(k1_res,k2_res))
    k1_flow=np.reshape(k1_flow,(k1_res,k2_res))
    k2_flow=np.reshape(k2_flow,(k1_res,k2_res))
    GG_temp=np.reshape(GG_temp,(k1_res,k2_res))

    plt.rc('text', usetex=True)
    fontsize=25
    labelsize=25
    
    fig = plt.figure(figsize=(15,12))
    ax1 = plt.subplot2grid((7,2), (0, 0), colspan=2, rowspan=2)
    ax1.plot(t,ord_par[0:], color='#637adeff',alpha=0.8,rasterized=True)
    ax1.set_xticks([0,200])
    ax1.set_yticks([0,1])
    ax1.tick_params(labelsize=labelsize)
    ax1.set_xlabel('\\textnormal{time in s}', fontsize=fontsize,va='bottom')
    ax1.set_ylabel('\\textnormal{R}', fontsize=fontsize, rotation=0,va='center', ha='left')
    ax1.set_xlim(right=200)
    ax1.text(0.01,0.8,'\\bf{a}',transform=ax1.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))
    ax1.add_patch(plt.Rectangle((70/200, 0.02), 0.02, 0.96,linewidth=2.5, ec='k',facecolor='none', alpha=0.5, transform=ax1.transAxes,zorder=10))
    ax1.add_patch(plt.Rectangle((158/200, 0.02), 0.02, 0.96,linewidth=2.5, ec='k',facecolor='none', alpha=0.5, transform=ax1.transAxes,zorder=10))
    
    ax2 = plt.subplot2grid((7,2), (2, 0), colspan=1)
    ax2.plot(t[((t>=70) & (t<=72))],ord_par[((t>=70) & (t<=72))], color='#637adeff', linewidth=3,alpha=0.8,rasterized=True)
    ax2.set_xticks([70,72])
    ax2.set_yticks([0,1])
    ax2.tick_params(labelsize=labelsize)
    ax2.set_xlabel('\\textnormal{time in s}', fontsize=fontsize, va='bottom')
    ax2.set_ylabel('\\textnormal{R}', fontsize=fontsize, rotation=0,va='center', ha='left')
    ax2.set_xlim(right=72)
    ax2.text(0.02,0.4,'\\bf{b}',transform=ax2.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))    
    
    ax3 = plt.subplot2grid((7,2), (2, 1),sharey=ax2, colspan=1)
    ax3.plot(t[((t>=156) & (t<=160))],ord_par[((t>=156) & (t<=160))], color='#637adeff', linewidth=3,alpha=0.8,rasterized=True)
    ax3.set_xticks([156,160])
    ax3.set_yticks([0,1])
    ax3.tick_params(labelsize=labelsize)
    ax3.set_xlabel('\\textnormal{time in s}', fontsize=fontsize, va='bottom')
    ax3.set_xlim(right=160)
    ax3.text(0.02,0.4,'\\bf{c}',transform=ax3.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))       
    
    ax4 = plt.subplot2grid((7,2), (3, 0), rowspan=3)
    ax4.contourf(k1_grid,k2_grid,GG_temp,1, colors=['#e0ecf4ff','#8856a733'])
    ax4.streamplot(k1_grid,k2_grid,k1_flow,k2_flow,density=(10,1),color='black', arrowsize=2)
    ax4.plot(k1_arr,k2_arr,color='#008000ff', linewidth=2, rasterized=True)
    ax4.contour(k1_grid,k2_grid,GG_temp,[1], colors='r', linewidths=2)
    ax4.set_xticks([0,0.2])
    ax4.set_yticks([0.6,1.3])
    ax4.tick_params(labelsize=labelsize)
    ax4.set_xlabel('$\\kappa_1$', fontsize=fontsize, va='bottom')
    ax4.set_ylabel('$\\kappa_2$', fontsize=fontsize, rotation=0,va='center', ha='left')
    ax4.set_xlim([0.0,0.25])
    ax4.set_ylim([0.5,1.3])
    ax4.text(0.02,0.85,'\\bf{d}',transform=ax4.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))

    cmap = colors.ListedColormap(['#0000ff33', '#ffcc0033','#ff000033','white'])
    bounds=[0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    ax5 = plt.subplot2grid((7,2), (3, 1), rowspan=3)
    ax5.pcolormesh(tau_bif,gamma_bif,GG_bif,cmap=cmap, antialiased=True,vmin=0, shading='auto')
    ax5.set_xticks([0.15,0.3])
    ax5.set_yticks([0.1,1.0])
    ax5.tick_params(labelsize=labelsize)
    ax5.set_xlabel('$\\tau_1$', fontsize=fontsize, va='bottom')
    ax5.set_ylabel('$\\gamma$', fontsize=fontsize, rotation=0, va='center',ha='left')
    ax5.text(0.02,0.85,'\\bf{e}',transform=ax5.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))
    ax5.text(0.383,0.455,'recurrent \n'+'synch.',transform=ax5.transAxes, fontsize=34, rotation=0)
    
    
    fig.tight_layout()
    fig.savefig('figure_3.svg', dpi=300)
    fig.savefig('figure_3.png',dpi=300)
fig_2HHN_plot( k1, k2, k1_flow, k2_flow, GG, k1_arr, k2_arr, N1_spikes, N2_spikes, t_update, 0, 200000)
