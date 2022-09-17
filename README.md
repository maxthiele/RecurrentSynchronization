# Recurrent Synchronization

Code repository accompanying the publication: "*Asymmetric Adaptivity induces Recurrent Synchronization in Complex Networks*".  
This project contains the code to produce the data for figures 2, 6 and 8 as well as the figures themselves. The directory ``` Simulation_data ``` only contains folders for the different figures with the data being saved in the corresponding folders. All duration times mentioned in the following are based on an i7-950 machine.

## Requirements

The required packages are listed in ``` requirements.txt ```. To install these execute ``` pip install -r requirements.txt ```.

## Figure 2

To obtain the data used in Figure 6 execute ``` python fig_2_data.py chosen ```. To start with random initial conditions execute ``` python fig_2_data.py random ```. The panels c-g of Figure 2 can be recreated using ``` python fig_2_plot.py``` afterwards.

## Figure 6

To obtain the grid used to produce the reduced flow in panel d execute ``` python fig_6_static_grid_main.py``` (**SLOW!** Multiple days.). The data for the bifurcation diagram in panel e can be produced using ``` python fig_6_bif_diagram_data_main.py ``` (**SLOW!** 18 hrs.). To obtain the time series data used in Figure 3 execute ``` python fig_6_data.py ```. Figure 6 can be created using ``` python fig_6_plot.py``` afterwards.

## Figure 8

The areas of recurrent synchronization in panel a can be produced by executing ``` python fig_8_bif_diag.py```. The time series and trajectories shown in the remaining panel can be obtained using ``` python fig_8_data.py```. Afterwards, the figure can be created by executing ``` python fig_8_plot.py ```.
