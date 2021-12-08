# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:19:05 2021

@author: Max
"""


import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from timeit import default_timer as timer
import pickle
import matplotlib.cm as cm
import os
from scipy.ndimage import uniform_filter1d, zoom
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import uniform_filter1d

#%%

k=np.load(r'Simulation_data/Figure_2/k.npy')
spiketimes=np.load(r'Simulation_data/Figure_2/spiketimes.npy')
order=np.load(r'Simulation_data/Figure_2/order.npy')
t=np.load(r'Simulation_data/Figure_2/t.npy')
spikerate=np.load(r'Simulation_data/Figure_2/spikerate.npy')
spikerate1=np.load(r'Simulation_data/Figure_2/spikerate_pop_1.npy')
spikerate2=np.load(r'Simulation_data/Figure_2/spikerate_pop_2.npy')
t_int=np.load(r'Simulation_data/Figure_2/t_int.npy')



#%%
t_anf_whole=000000
t_end_whole=300000

t_anf_inset=100000
t_end_inset=101000

t_anf_inset_ratio=(t_anf_inset+t_anf_whole)/t_end_whole

n1=100
n2=100
N=n1+n2
plt.rc('text', usetex=True)

fig = plt.figure(figsize=(15,12))
ax4=plt.subplot2grid((3,2), (1, 0), colspan=2, rowspan=1)
ax4.plot(t_int/1000, spikerate1,'s-', color='#d95f02ff',markersize=10, linewidth=2,rasterized='True')
ax4.plot(t_int/1000, spikerate2,'s-', color='#1b9e77ff',markersize=10, linewidth=2,rasterized='True')
ax4.plot(t_int/1000, spikerate,'s-', color='#637adeff',markersize=10, linewidth=2,rasterized='True')
ax4.set_yticks([190,230])
ax4.set_xticks([0,300])
ax4.tick_params(labelsize=25)
ax4.set_xlim(right=300)
ax4.set_ylabel('\\textnormal{firing density}',va='top', fontsize=25)
ax4.set_xlabel('\\textnormal{time in s}', fontsize=25,va='bottom')
ax4.text(0.01,0.80,'\\bf{e}',transform=ax4.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))







ax1 = plt.subplot2grid((3,2), (0, 0), colspan=2, rowspan=1, sharex=ax4)
ax1.plot((t[((t>=t_anf_whole) & (t<=t_end_whole))]-t_anf_whole)/1000,order[((t>=t_anf_whole) & (t<=t_end_whole))]
         , color='#637adeff',zorder=1, alpha=0.8, rasterized='True')

ax1.set_yticks([0,1])
ax1.tick_params(labelsize=25)

ax1.set_ylabel('\\textnormal{R}', fontsize=25, rotation=0,va='center')

ax1.set_xlim(right=300)
ax1.set_zorder(1)
plt.setp(ax1.get_xticklabels(), visible=False)


ax1.text(0.01,0.80,'\\bf{c}',transform=ax1.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))


axins = inset_axes(ax1, width="100%", height="100%", bbox_to_anchor=(0.02, 0.1, .3, .5), bbox_transform=ax1.transAxes, loc=3)
axins.plot(t[((t>=t_anf_inset) & (t<=t_end_inset))]/1000,order[((t>=t_anf_inset) & (t<=t_end_inset))],linewidth=3, color='#637adeff')
axins.patch.set_alpha(0.0)
axins.set_xticks([t_anf_inset/1000,t_end_inset/1000])
axins.set_yticks([0,1])
axins.tick_params(labelsize=25)
axins.text(0.02,0.65,'\\bf{d}',transform=axins.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))
axins.set_zorder(10)
ax1.add_patch(plt.Rectangle((0.001, 0.001), .33, .65,ec='none',facecolor='white', alpha=0.9, transform=ax1.transAxes,zorder=10))
ax1.add_patch(plt.Rectangle((t_anf_inset_ratio+0.02, 0.02), 0.01, 0.96,linewidth=3, ec='k',facecolor='none', alpha=0.5, transform=ax1.transAxes,zorder=10))






ax2 = plt.subplot2grid((3,2), (2, 0), colspan=1, rowspan=1)
t_anf_synch=130000
t_end_synch=130100
index=np.arange(N)+1
for i in range(N):
    temp=spiketimes[i,:]
    temp=temp[((temp>t_anf_synch) & (temp<t_end_synch))]
    temp1=np.ones(len(temp))*index[i]
    if i<n1:
        ax2.plot(temp/1000, temp1,'.',color='#d95f02ff',markersize=10,rasterized='True')
    else:
        ax2.plot(temp/1000, temp1,'.',color='#1b9e77ff',markersize=10,rasterized='True')

ax2.ticklabel_format(useOffset=False)
ax2.set_xlabel('\\textnormal{time in s}',fontsize=25,va='bottom')
ax2.set_ylabel('\\textnormal{index}',fontsize=25)
ax2.yaxis.set_label_coords(-0.05, 0.5)
ax2.set_xticks([t_anf_synch/1000,t_end_synch/1000])
ax2.set_yticks([1,200])
ax2.tick_params(labelsize=25)
ax2.text(0.02,0.8,'\\bf{f}',transform=ax2.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))

ax3 = plt.subplot2grid((3,2), (2, 1), colspan=1, rowspan=1)
t_anf_asynch=230000
t_end_asynch=230100
index=np.arange(N)+1
for i in range(N):
    temp=spiketimes[i,:]
    temp=temp[((temp>t_anf_asynch) & (temp<t_end_asynch))]
    temp1=np.ones(len(temp))*index[i]
    if i<n1:
        ax3.plot(temp/1000, temp1,'.',color='#d95f02ff',markersize=10,rasterized='True')
    else:
        ax3.plot(temp/1000, temp1,'.',color='#1b9e77ff',markersize=10,rasterized='True')

ax3.ticklabel_format(useOffset=False)
ax3.set_xlabel('\\textnormal{time in s}',fontsize=25,va='bottom')

ax3.yaxis.set_label_coords(-0.05, 0.5)
ax3.set_xticks([t_anf_asynch/1000,t_end_asynch/1000])
ax3.set_yticks([1,200])
ax3.tick_params(labelsize=25)
ax3.text(0.02,0.83,'\\bf{g}',transform=ax3.transAxes, fontsize=50,bbox=dict(facecolor='white',edgecolor='none'))




fig.tight_layout()

fig.savefig('fig HHN network python firing density.svg',dpi=300)