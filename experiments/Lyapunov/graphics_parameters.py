"""
Graphics parameters for the Lyapunov experiments.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% GRAPHICS PARAMETERS
import matplotlib.pyplot as plt

# Save original parameters
original_params = plt.rcParams.copy()

# Custom parameters
plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 125
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['figure.autolayout'] = True

# Save custom parameters
custom_params = plt.rcParams.copy()

# Parameters for saving the plots
path = 'figures/'
do_save = True

# Make directory if it does not exist
import os
if do_save:
    if not os.path.exists(path):
        os.makedirs(path)


