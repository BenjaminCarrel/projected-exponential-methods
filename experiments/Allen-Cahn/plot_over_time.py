"""
File for plotting the solution over over time.

Author: Benjamin Carrel, University of Geneva, 2023
"""

#%% IMPORTATIONS
# from graphics_parameters import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy import ndimage
from datetime import datetime
from problems import make_allen_cahn
from low_rank_toolbox import SVD, LowRankMatrix
from matrix_ode_toolbox.integrate import solve_matrix_ivp
import numpy as np
from matrix_ode_toolbox.dlra import solve_dlra


#%% SETUP THE ODE
size = 256
t_span = (0, 10)
make_ode = make_allen_cahn
ode, X0 = make_ode(size)
nb_steps = 100


# Print the ode defined in setup_problem.py
print(ode)
if not isinstance(X0, LowRankMatrix):
    X0 = SVD.reduced_svd(X0)
print('Dimensions: ', X0.shape)

#%% Reference solution
ts = np.linspace(*t_span, nb_steps+1)
ref_sol = solve_matrix_ivp(ode, t_span, X0, solver="scipy", t_eval=ts, monitor=True, scipy_method='RK45', atol=1e-8, rtol=1e-8)
Xs_ref = ref_sol.todense()
print('Time taken by the reference solver (scipy RK45 with tol=1e-8): {:.2f} s'.format(sum(ref_sol.computation_time)))

#%% Plot at different times
time_to_plot = [0, 3, 5, 7, 10]
nb_plots = len(time_to_plot)
fig, axs = plt.subplots(nb_plots, 1, figsize=(4, 24), sharex=True, sharey=True)
fig.suptitle(f'Reference solution', fontsize=24)

for i, t in enumerate(time_to_plot):
    frame_number = np.argmin(np.abs(ref_sol.ts - t))
    rotated_img = ndimage.rotate(Xs_ref[frame_number], 0)
    im = axs[i].imshow(rotated_img, cmap='jet', extent=[0, 2*np.pi, 0, 2*np.pi], interpolation='bilinear')
    axs[i].set_title('t = {:.2f}'.format(t), fontsize=24)
    axs[i].set_xticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    axs[i].set_yticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    # axs[i].set_xlabel(r'$x$')
    # axs[i].set_ylabel(r'$y$')
    axs[i].grid(False)

# Move the color bar outside of the plots
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("bottom", size="5%", pad=0.5)
fig.colorbar(im, cax=cax, orientation='horizontal')

plt.tight_layout()

# Save it with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
fig.savefig(f'figures/reference_solution_over_time_{timestamp}.pdf', bbox_inches='tight')


plt.show()

# %% DLRA solution

# Initial value
rank = 2
Y0 = SVD.truncated_svd(X0, rank)

# Solver parameters
k = 2 # Something fun happen when k=1
poles = np.repeat(k / np.sqrt(2), k)
krylov_kwargs = {'size': k, 'kind': 'rational', 'poles': poles}
order = 1
kwargs = {'order': order, 'krylov_kwargs': krylov_kwargs, 'use_closed_form': False}

# Solve
dlra_sol = solve_dlra(ode, t_span, Y0, dlra_solver="PERK", solver_kwargs=kwargs, t_eval=ts, monitor=True)
print('Time taken by projected exponential Runge (rank=10): {:.2f} s'.format(sum(dlra_sol.computation_time)))
Ys = dlra_sol.todense()

#%% Plot at different times
fig, axs = plt.subplots(nb_plots, 1, figsize=(4, 24), sharex=True, sharey=True)
fig.suptitle(fr'Proj. exp. Euler ($r={rank}$)', fontsize=24)

for i, t in enumerate(time_to_plot):
    frame_number = np.argmin(np.abs(dlra_sol.ts - t))
    rotated_img = ndimage.rotate(Ys[frame_number], 0)
    im = axs[i].imshow(rotated_img, cmap='jet', extent=[0, 2*np.pi, 0, 2*np.pi], interpolation='bilinear')
    axs[i].set_title('t = {:.2f}'.format(t), fontsize=24)
    axs[i].set_xticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    axs[i].set_yticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    # axs[i].set_xlabel(r'$x$')
    # axs[i].set_ylabel(r'$y$')
    axs[i].grid(False)

# Move the color bar outside of the plots
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("bottom", size="5%", pad=0.5)
fig.colorbar(im, cax=cax, orientation='horizontal')

plt.tight_layout()

# Save it with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
fig.savefig(f'figures/perk{order}_solution_over_time_{timestamp}.pdf', bbox_inches='tight')

plt.show()


#%% Plot the error at different times
fig, axs = plt.subplots(nb_plots, 1, figsize=(4, 24), sharex=True, sharey=True)
fig.suptitle(f'Difference', fontsize=24)
for i, t in enumerate(time_to_plot):
    frame_number = np.argmin(np.abs(dlra_sol.ts - t))
    error = (Ys[frame_number] - Xs_ref[frame_number]) / np.linalg.norm(Xs_ref[frame_number], 'fro')
    rotated_img = ndimage.rotate(error, 0)
    im = axs[i].imshow(rotated_img, cmap='jet', extent=[0, 2*np.pi, 0, 2*np.pi], interpolation='bilinear')
    axs[i].set_title('t = {:.2f}'.format(t), fontsize=24)
    axs[i].set_xticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    axs[i].set_yticks([0, np.pi, 2*np.pi], [0, r'$\pi$', r'$2\pi$'], fontsize=24)
    # axs[i].set_xlabel(r'$x$')
    # axs[i].set_ylabel(r'$y$')
    axs[i].grid(False)

# Move the color bar outside of the plots
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("bottom", size="5%", pad=0.5)
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

plt.tight_layout()

# Save the figure with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
fig.savefig(f'figures/error_over_time_{timestamp}.pdf', bbox_inches='tight')

# Display the plot
plt.show()



# %%
