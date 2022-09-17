#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')


#%%

eps=0.0001
alpha=np.pi*0.25
omega=0.1
res=1000
stepsize=0.01
mstep=0.001
timeunits=200000
timeunits_rev=200000
area_fold=0.5
area_val=0.5
area=[-area_val,area_val,-area_val,area_val]





#%%
y_LC_upper=np.load(r'Simulation_data/Figure_4/y_LC_upper.npy')[::10,0]
y_LC_lower=np.load(r'Simulation_data/Figure_4/y_LC_lower.npy')[::10,0]
y_00_lower=np.load(r'Simulation_data/Figure_4/y_00_lower.npy')[::10,0]
t=np.load(r'Simulation_data/Figure_4/t.npy')[::10]
y_red_LC_upper=np.load(r'Simulation_data/Figure_4/y_red_LC_upper.npy')
y1_red_upper=np.load(r'Simulation_data/Figure_4/y1_red_upper.npy')
y2_red_upper=np.load(r'Simulation_data/Figure_4/y2_red_upper.npy')

y_red_LC_lower=np.load(r'Simulation_data/Figure_4/y_red_LC_lower.npy')
y1_red_lower=np.load(r'Simulation_data/Figure_4/y1_red_lower.npy')
y2_red_lower=np.load(r'Simulation_data/Figure_4/y2_red_lower.npy')
y3_red_lower=np.load(r'Simulation_data/Figure_4/y3_red_lower.npy')
y_unstable_LC=np.load(r'Simulation_data/Figure_4/y_unstable_LC.npy')
y00_red_lower=np.load(r'Simulation_data/Figure_4/y00_red_lower.npy')

a_bursting1=np.load(r'Simulation_data/Figure_4/bursting_data_a_b_grid/a_burst1.npy')
b_bursting1=np.load(r'Simulation_data/Figure_4/bursting_data_a_b_grid/b_burst1.npy')
a_bursting2=np.load(r'Simulation_data/Figure_4/bursting_data_a_b_grid/a_burst2.npy')
b_bursting2=np.load(r'Simulation_data/Figure_4/bursting_data_a_b_grid/b_burst2.npy')


a_ZZ_line=np.linspace(-1,1,10000)
b_ZZ_line=(-a_ZZ_line*np.cos(alpha)+2*omega)/(-a_ZZ_line/omega*np.cos(alpha)*np.sin(alpha)+np.sin(alpha))

a_ZZ_line_2=np.linspace(-1,1,10000)
b_ZZ_line_2=(-a_ZZ_line_2*np.cos(alpha)+4*omega)/(np.sin(alpha))
a_ZZ_line_2_cond=a_ZZ_line_2[np.where(a_ZZ_line_2*b_ZZ_line_2*np.sin(alpha)*np.cos(alpha)/(omega)-a_ZZ_line_2*np.cos(alpha)-b_ZZ_line_2*np.sin(alpha)+2*omega>0)]
b_ZZ_line_2_cond=b_ZZ_line_2[np.where(a_ZZ_line_2*b_ZZ_line_2*np.sin(alpha)*np.cos(alpha)/(omega)-a_ZZ_line_2*np.cos(alpha)-b_ZZ_line_2*np.sin(alpha)+2*omega>0)]

b_ZZ_line_temp=b_ZZ_line[(((a_ZZ_line>=np.min(a_ZZ_line_2_cond)) & (b_ZZ_line<=np.min(b_ZZ_line_2_cond))) | ((a_ZZ_line<=np.min(a_ZZ_line_2_cond)) & (b_ZZ_line>=np.min(b_ZZ_line_2_cond))))]
a_ZZ_line_temp=a_ZZ_line[(((a_ZZ_line>=np.min(a_ZZ_line_2_cond)) & (b_ZZ_line<=np.min(b_ZZ_line_2_cond))) | ((a_ZZ_line<=np.min(a_ZZ_line_2_cond)) & (b_ZZ_line>=np.min(b_ZZ_line_2_cond))))]
a_ZZ_line=a_ZZ_line_temp
b_ZZ_line=b_ZZ_line_temp





#%%
mpl.rcParams['axes.unicode_minus'] = False   
def calc_and_plots(): 
    
    #Foldline
    k1_fold1=np.linspace(-area_fold,area_fold,res*100)
    k1_fold2=np.linspace(-area_fold,area_fold,res*100)
    k2_fold1=-k1_fold1*np.cos(2*alpha)+np.sqrt(k1_fold1**2*(np.cos(2*alpha)**2-1)+omega**2)
    k2_fold2=-k1_fold1*np.cos(2*alpha)-np.sqrt(k1_fold1**2*(np.cos(2*alpha)**2-1)+omega**2)
    k1_fold1=k1_fold1[~np.isnan(k2_fold1)]
    k1_fold2=k1_fold2[~np.isnan(k2_fold2)]
    k2_fold1=k2_fold1[~np.isnan(k2_fold1)]
    k2_fold2=k2_fold2[~np.isnan(k2_fold2)]
    k1_fold=np.append(k1_fold1,k1_fold1)
    k2_fold=np.append(k2_fold1,k2_fold2)
    

    plt.rc('text', usetex=True)
    fontsize=35
    labelsize=30
    fig=plt.figure(figsize=(21,8))
    
    
# order parameter LC1         
    ax1 = plt.subplot2grid((4,7), (0, 5), colspan=2, rowspan=2)
    y=np.abs(np.exp(1j*y_LC_upper[:])+1)/2
    ax1.plot(t,y,'#008000ff')
    ax1.set_xticks([0,200000])
    ax1.set_yticks([0,1])
    ax1.tick_params(labelsize=labelsize)
    ax1.set_xlabel('\\textnormal{time}', fontsize=fontsize, va='bottom')
    ax1.set_ylabel('\\textnormal{R}', fontsize=fontsize, rotation=0,va='center',ha='left')
    ax1.text(0.02,0.85,'\\bf{c}',transform=ax1.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
    ax1.text(0.41,0.25,'anti-phase',transform=ax1.transAxes,fontsize=40, rotation=90)
    ax1.text(0.65,0.25,'in-phase',transform=ax1.transAxes,fontsize=40, rotation=90)
    ax1.text(0.3,0.12,'running phase', color='white',transform=ax1.transAxes,fontsize=35, rotation=90)
    


# order parameter LC2
    ax2 = plt.subplot2grid((4,7), (2, 5), colspan=2, rowspan=1)
    y=np.abs(np.exp(1j*y_LC_lower[:])+1)/2
    ax2.plot(t,y,'#008000ff')
    ax2.set_xticks([0,200000])
    ax2.set_yticks([0,1])
    ax2.tick_params(labelsize=labelsize)
    ax2.set_xlabel('\\textnormal{time}', fontsize=fontsize, va='bottom')
    ax2.set_ylabel('\\textnormal{R}', fontsize=fontsize, rotation=0,va='center',ha='left')
    ax2.text(0.02,0.65,'\\bf{e}',transform=ax2.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
    
    
    
## order parameter FP    
    ax3 = plt.subplot2grid((4,7), (3, 5), colspan=2, rowspan=1)
    y=np.abs(np.exp(1j*y_00_lower[:])+1)/2
    ax3.plot(t[:10000],y[:10000],color='#aa2704ff')
    ax3.set_xticks([0,1000])
    ax3.set_yticks([0,1])
    ax3.tick_params(labelsize=labelsize)
    ax3.set_xlabel('\\textnormal{time}', fontsize=fontsize, va='bottom')
    ax3.set_ylabel('\\textnormal{R}', fontsize=fontsize, rotation=0,va='center',ha='left')
    ax3.text(0.02,0.65,'\\bf{f}',transform=ax3.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
 
    
 
# phase diagram 1
    ax4 = plt.subplot2grid((4,7), (0, 3), colspan=2, rowspan=2)
    ax4.plot(k1_fold1,k2_fold1,'r')
    ax4.plot(k1_fold2,k2_fold2,'r')
    ax4.fill_between(k1_fold1,k2_fold1,k2_fold2,color='#e0ecf4ff')
    ax4.fill_between(k1_fold1,k2_fold1,0.3,color='#8856a733', edgecolor='none')
    ax4.fill_between(k1_fold1,k2_fold2,-0.3,color='#8856a733', edgecolor='none')
    ax4.fill_between([np.min(k1_fold1),-0.3],-0.3,0.3,color='#8856a733', edgecolor='none')
    ax4.fill_between([np.max(k1_fold1),0.3],0.3,-0.3,color='#8856a733', edgecolor='none')
    ax4.plot(y1_red_upper[0,:],y1_red_upper[1,:],'k')
    ax4.plot(y2_red_upper[0,:],y2_red_upper[1,:],'k')
    ax4.plot(y_red_LC_upper[0,:],y_red_LC_upper[1,:],'#008000ff')
    ax4.scatter(0,0,ec='#aa2704ff',fc='none', s=50,zorder=10)
    ax4.set_xlim([-0.15,0.15])
    ax4.set_ylim([-0.15,0.15])
    ax4.set_xticks([-0.15,0.15])
    ax4.set_yticks([-0.15,0.15])
    ax4.tick_params(labelsize=labelsize)
    ax4.set_xlabel('$\\kappa_1$', fontsize=fontsize, va='bottom')
    ax4.set_ylabel('$\\kappa_2$', fontsize=fontsize, rotation=0)
    ax4.yaxis.set_label_coords(-0.07, 0.5)
    ax4.text(0.02,0.85,'\\bf{b}',transform=ax4.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
    
    
# phase diagram 2    
    ax5 = plt.subplot2grid((4,7), (2, 3), colspan=2, rowspan=2)
    ax5.plot(k1_fold1,k2_fold1,'r')
    ax5.plot(k1_fold2,k2_fold2,'r')
    ax5.fill_between(k1_fold1,k2_fold1,k2_fold2,color='#e0ecf4ff')
    ax5.fill_between(k1_fold1,k2_fold1,0.3,color='#8856a733', edgecolor='none')
    ax5.fill_between(k1_fold1,k2_fold2,-0.3,color='#8856a733', edgecolor='none')
    ax5.fill_between([np.min(k1_fold1),-0.3],-0.3,0.3,color='#8856a733', edgecolor='none')
    ax5.fill_between([np.max(k1_fold1),0.3],-0.3,0.3,color='#8856a733', edgecolor='none')
    ax5.plot(y1_red_lower[0,:],y1_red_lower[1,:],'k')
    ax5.plot(y2_red_lower[0,:],y2_red_lower[1,:],'k')
    ax5.plot(y3_red_lower[0,:],y3_red_lower[1,:],'k')
    ax5.plot(y00_red_lower[0,:],y00_red_lower[1,:],'k')
    ax5.plot(y_red_LC_lower[0,:],y_red_LC_lower[1,:],'#008000ff')
    ax5.plot(y_unstable_LC[0,:],y_unstable_LC[1,:],'#0000ffff')
    ax5.scatter(0,0,ec='#aa2704ff', fc='#aa2704ff', s=50,zorder=10)
    ax5.set_xlim([-0.15,0.15])
    ax5.set_ylim([-0.15,0.15])
    ax5.set_xticks([-0.15,0.15])
    ax5.set_yticks([-0.15,0.15])
    ax5.tick_params(labelsize=labelsize)
    ax5.set_xlabel('$\\kappa_1$', fontsize=fontsize, va='bottom')
    ax5.set_ylabel('$\\kappa_2$', fontsize=fontsize, rotation=0)
    ax5.yaxis.set_label_coords(-0.07, 0.5)
    ax5.text(0.02,0.85,'\\bf{d}',transform=ax5.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
    
    
# bif diagram    
    ax6 = plt.subplot2grid((4,7), (0, 0), colspan=3, rowspan=4)
    ax6.plot(a_bursting1,b_bursting1,color='#ffcc0033')
    ax6.plot(a_bursting2,b_bursting2,color='#ffcc0033')
    ax6.fill_betweenx(b_bursting1,a_bursting1,0.5,color='#ffcc0033', ec='none')
    ax6.fill_between(a_bursting2,b_bursting2,0.5,color='#ffcc0033', ec='none')
    ax6.scatter(a_ZZ_line,b_ZZ_line,color='#0000004d', s=1)
    ax6.scatter(a_ZZ_line_2_cond,b_ZZ_line_2_cond,color='#0000001a', s=1)
    ax6.plot(np.linspace(-0.5,0,100),np.linspace(0.5,0,100),'--', dashes=(5,10),color= 'red')
    ax6.set_xlim([0,0.5])
    ax6.set_ylim([0,0.5])
    ax6.set_xticks([0,0.5])
    ax6.set_yticks([0,0.5])
    ax6.tick_params(labelsize=labelsize)
    ax6.set_xlabel('$a$', fontsize=fontsize, va='bottom')
    ax6.set_ylabel('$b$', fontsize=fontsize, rotation=0,va='center',ha='left')
    ax6.text(0.9,0.25,'\\bf{A}',transform=ax6.transAxes, fontsize=50)
    ax6.text(0.7,0.2,'\\bf{B}',transform=ax6.transAxes, fontsize=50)
    ax6.text(0.01,0.925,'\\bf{a}',transform=ax6.transAxes, fontsize=40,bbox=dict(facecolor='white',edgecolor='none'))
    
    
    fig.tight_layout(w_pad=-2, h_pad=-1)
    fig.savefig('figure_4.svg')
calc_and_plots()



