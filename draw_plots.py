from typing import cast
from PIL.ImageQt import rgb
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from iminuit import Minuit, cost

from hdf5_utils import read_data
from autocorrelation import tau_int_sokal, tau_int_blocking, autocorrelation
from mcmc_utils import metropolis_ising
from operators import energy, magnetization
from analytical_curves import plot_analytical_curve


def graph_magnetization_convergence(sources, filename = 'magnetization_convergence.png'):
    '''
    Plots magnetizations functions of T for different Ns and saves the image.
    A convergence should be observed as N increases (for dim = 1 this doesn't mathematically converge, but for dim = 2 and 3 it should).

    - `sources` is a list of file paths to HDF5 files containing the simulation data:
        this function should only extract the already computed magnetization functions
        and should be only called after said data has been extracted.
    - `filename` is the output file path.

    Returns the saved file path.
    '''

    if len(sources) <= 0:
        raise ValueError('`sources` must contain at least one file path')

    # Sorting by N
    def get_n_from_file(filepath):
        with h5py.File(filepath, 'r') as f:
            group_name = list(f.keys())[0]
            return int(group_name.split('_')[3])

    sources = sorted(sources, key = get_n_from_file)
    cmap = plt.get_cmap('viridis_r')
    L = len(sources)

    plt.figure(figsize = (12, 7))

    for i, source in enumerate(sources):
        with h5py.File(source, 'r') as f:
            # Extract dim and N from the group name
            group_name = list(f.keys())[0]

            temperatures = np.array(f[f"{group_name}/temperatures"])
            magnetizations = np.array(f[f"{group_name}/magnetizations"])
            errors = np.array(f[f"{group_name}/magnetization_errors"])

            def norm(x):
                return np.power(x, 2)
            color_val = norm(i + 1) / norm(L) if L > 1 else 0.0

            plt.errorbar(
                temperatures, 
                magnetizations, 
                yerr = errors, 
                color = mcolors.to_rgba(cmap(color_val), alpha = .4),
                ecolor = mcolors.to_rgba(cmap(color_val), alpha = 1),
                capsize = 3,
                marker = '.', 
                label = f"N = {group_name.split('_')[3]}",
                # linestyle = (0, (1, 1))
                )

    # Plot analytical curve
    # with h5py.File(source, 'r') as f:
    #     temps = np.array(f[f"{list(f.keys())[0]}/temperatures"])
    #     t = np.linspace(temps[5], temps[-5], 100)
    # plot_analytical_curve(dim = int(sources[0].split('_')[2]), t = t)

    plt.xlabel('Temperature (T)')
    plt.ylabel('Magnetization')
    plt.title('Magnetization vs Temperature for different N')
    plt.grid()
    plt.legend()
    plt.savefig(filename, format='pdf', bbox_inches = 'tight')
    plt.close()


def metropolis_ising_graphics(initial_states, T: float, steps: int, seed: int):
    '''
    Given:
        A dictionary of initial states of an ising model
        A fixed temperature T
        Number of steps to take (in the markov process)
    
    Saves an image containing:
    A graph representing the evolution of the energy of the system over time for each initial state.
    The distribution of the energy of the system is also plotted at the end of the simulation.
    '''

    # fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize = (9, 9), sharex = True, gridspec_kw = {'height_ratios': [1, 1]})
    fig, ax_down = plt.subplots(1, 1, figsize = (9, 5))
    first = True

    colors = [
        (253/255, 231/255, 37/255),  # yellow
        (53/255, 183/255, 121/255),  # green
        (72/255, 40/255, 120/255)    # purple
    ]

    for i, initial_state in enumerate(initial_states.keys()):
        print(f"Running Metropolis-Hastings for initial state: {initial_state} at T = {T} for {steps} steps...")
        models_samples = metropolis_ising(initial_states[initial_state], T, steps, seed + i)
        energies = np.array([energy(model) for model in models_samples])

        current_color = colors[i % len(colors)]

        if first:
            first = False

            tau_int = tau_int_sokal(energies)[0]
            ax_down.axvspan(
                0,
                20 * tau_int,
                color='gray',
                alpha=0.3,
                zorder=0,
                label=f'Thermalization time'
            )

        #     def f_pdf(E, mu, sigma):
        #         return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((E - mu) / sigma)**2)

        #     c = cost.UnbinnedNLL(energies, f_pdf)
        #     m = Minuit(c, mu = -200, sigma = 10)
        #     print(m.migrad())

        #     x = np.linspace(energies.min(), energies.max(), 100)
        #     ax_up.plot(x, f_pdf(x, mu=m.values["mu"], sigma=m.values["sigma"]), label='Normal', color="black", linestyle="dotted")

        step = 1
        ax_down.plot(
            range(0, len(energies), step),
            energies[::step],
            color=current_color,
            alpha=1,
            label=f'Initial state: {initial_state}'
        )

        # ax_up.hist(
        #     energies,
        #     bins=(1 + int(np.log2(len(energies)))),
        #     edgecolor=current_color,
        #     linewidth=1,
        #     linestyle='solid',
        #     facecolor=current_color,
        #     density=True,
        #     histtype='stepfilled',
        #     alpha = 1,
        #     label=f'Initial state: {initial_state}'
        # )

    
    
    ax_down.set_ylabel('Energy')
    ax_down.set_xlabel('Time')
    ax_down.set_title(f'Energy Evolution: T = {T}, Shape = {models_samples.shape}')
    ax_down.legend(loc='lower right')
    ax_down.grid()
    ax_down.set_axisbelow(True)
    
    # ax_up.set_ylabel('Probability Density')
    # ax_up.set_title(f'Energy Distribution: T = {T}, Shape = {models_samples.shape}')
    # ax_up.legend()
    # ax_up.grid()
    # ax_up.set_axisbelow(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1)

    plt.savefig(f"ED_{models_samples.shape}_{T}.pdf", format='pdf', bbox_inches = 'tight')
    plt.close()



def autocorrelation_graph(N, dim, data_file = "tmp.hdf5", filename = "autocorrelation.png", T_index = 30):
    '''
    Plots the autocorrelation function of an observable O as a function of time (steps) 
    given the raw data stored in an HDF5 file.
    '''

    LEN = 2000

    # Not using read_data() here to economize memory usage
    with h5py.File(data_file, "r") as file:
        temperatures = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/temperatures"]))
        filtered_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[T_index, :LEN])

    print(f"Filtered data shape: {filtered_data.shape} (T = {temperatures[T_index]:.2f})")

    times = np.arange(0, LEN, 1)
    acs = np.zeros_like(times, dtype = float)
    observables = np.array([magnetization(model) for model in filtered_data])

    for i, t in enumerate(times):
        acs[i] = autocorrelation(t, observables)

    plt.figure(figsize=(10, 6))
    plt.plot(times, acs, label='Autocorrelation function', color=(53 / 255, 183 / 255, 121 / 255))
    plt.plot(0, acs[0], label=f"Initial value: {acs[0]:.2f}", marker='x', markersize=8, color=(72 / 255, 40 / 255, 120 / 255))

    # x_min, x_max = 200, 10_000
    # y_min, y_max = -.3, .3
    # rect = Rectangle(
    #     xy=(x_min, y_min), 
    #     width=(x_max - x_min), 
    #     height=(y_max - y_min), 
    #     facecolor=(253 / 255, 231 / 255, 37 / 255),
    #     alpha=1,
    #     zorder=0
    # )
    # plt.gca().add_patch(rect)

    plt.xlabel('Time (steps)')
    plt.ylabel('Autocorrelation')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.title(f'Autocorrelation Function - T = {temperatures[T_index]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Autocorrelation graph saved to {filename}.")



def tau_int_graph(N, dim, data_file, filename = "tau_int.png"):
    '''
    Plots the integrated autocorrelation time (tau_int) with respect to magnetization
    as a function of temperature.k
    '''
    
    temperatures, data = read_data(data_file, N, dim)

    print(f"Filtered data shape: {data.shape}")

    observables = np.array([[magnetization(model) for model in models_at_T] for models_at_T in data])
    taus_sokal = np.zeros_like(temperatures)
    taus_sokal_errors = np.zeros_like(temperatures)
    taus_blocking = np.zeros_like(temperatures)
    taus_blocking_errors = np.zeros_like(temperatures)

    for i, T in enumerate(temperatures):
        print(f"Temperature: {T:.2f}")
        try:
            taus_sokal[i], M = tau_int_sokal(observables[i], c = 3.0)
            taus_sokal_errors[i] = taus_sokal[i] * np.sqrt((4 * M + 2) / len(observables[i])) if M > 0 else np.nan
            taus_blocking[i], Nb_arr, _ = tau_int_blocking(observables[i], c = 3.0)
            taus_blocking_errors[i] = (taus_blocking[i] + 0.5) * np.sqrt(2.0 / (Nb_arr[-1] - 1))
        except Exception as e:
            print(f"Warning: failed at T = {T:.2f}: {type(e).__name__}: {e}")
            taus_sokal[i] = np.nan
            taus_sokal_errors[i] = np.nan
            taus_blocking[i] = np.nan
            taus_blocking_errors[i] = np.nan

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize = (9, 9), sharex = True, gridspec_kw = {'height_ratios': [1, 1]})
    
    ax_up.errorbar(
        temperatures,
        taus_sokal,
        yerr = taus_sokal_errors,
        marker = "o",
        linestyle = "None",
        capsize = 3,
        ecolor = mcolors.to_rgba("#35B779", alpha = 1),
        color = mcolors.to_rgba("#35B779", alpha = 1),
        label = r"$\tau_{int}-sokal$"
        )
    ax_up.errorbar(
        temperatures + 0.02, # small offset for better visibility
        taus_blocking,
        yerr = taus_blocking_errors,
        marker = "o",
        linestyle = "None",
        capsize = 3,
        ecolor = mcolors.to_rgba("#482878", alpha = 1),
        color = mcolors.to_rgba("#482878", alpha = 1),
        label = r"$\tau_{int}-blocking$"
        )
    ax_up.set_xlabel('Temperature')
    ax_up.set_ylabel(r"$\tau_{int}$")
    ax_up.grid(True, which="both", ls="--")
    ax_up.set_title(r"$\tau_{int}$" + f" - N = {N}, dim = {dim}")
    ax_up.legend()

    ax_down.plot(
        temperatures,
        abs(taus_sokal - taus_blocking),
        marker = ".",
        label = r"$\tau_{int} errors$",
        linestyle = "solid",
        color = 'black'
        )
    ax_down.set_xlabel('Temperature')
    ax_down.set_ylabel(r"Difference")
    ax_down.grid(True, which="both", ls="--")
    ax_down.set_title(r"$\tau_{int}$" + f" - N = {N}, dim = {dim}")
    ax_down.legend()

    plt.savefig(filename, format='pdf', bbox_inches = 'tight')
    plt.close()

    
    # Save temperatures and taus to a text file
    data_to_save = np.column_stack((temperatures, taus_sokal, taus_blocking))




##################################################################################################
# Main execution block for testing and demonstration purposes
##################################################################################################

if __name__ == '__main__':


    #########################################################################################
    #########################################################################################
    # MAGNETIZATION VISUALIZATION EXAMPLE USAGE

    # Ns = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 500]
    # dims = [1]

    # Ns = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    Ns = [5, 10, 15, 16, 20, 30]
    dims = [2]
    
    # Ns = [4, 6, 8, 10, 15, 25]
    # dims = [3]

    sources = [r"dim_{dim}_N_{N}_data.hdf5".format(dim = dim, N = N) for N in Ns for dim in dims]
    graph_magnetization_convergence(sources, filename = f"magnetization_convergence_{dims[0]}D.png")




    #########################################################################################
    # Single file example usage
    # sources = [r"dim_2_N_50_data.hdf5"]
    # graph_magnetization_convergence(sources, filename = f"tmp.png")



    #########################################################################################
    #########################################################################################
    # MARKOV CHAIN VISUALIZATION EXAMPLE USAGE

    # N = (15, 15)
    # x1 = np.random.choice([-1, 1], size = N)
    # x2 = np.ones(N, dtype = np.int8)
    # x3 = - np.ones(N, dtype = np.int8)
    # SEED = 5

    # starting_configs = {
    #     "random": x1,
    #     "all_up": x2,
    #     "all_down": x3
    # }

    # metropolis_ising_graphics(starting_configs, T = 3.26, steps = 10_000, seed = SEED)



    #########################################################################################
    #########################################################################################
    # AUTOCORRELATION VISUALIZATION EXAMPLE USAGE

    # N = 20
    # dim = 2
    # data_file = "E:\\simulations_data\\" + f"dim_{dim}_N_{N}" + "_data.hdf5"
    # data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"
    # autocorrelation_graph(N, dim, data_file, filename = "autocorrelation.pdf", T_index = 30) 

    # with h5py.File(data_file, "r") as file:
    #     filtered_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"]))
    # observables = np.array([energy(model) for model in filtered_data])

    # tau_int_blocking_value = tau_int_blocking(observables, make_plot = True)



    #########################################################################################
    #########################################################################################
    # Tau_int VISUALIZATION EXAMPLE USAGE

    N = 10
    dim = 2
    data_file = f"dim_{dim}_N_{N}" + "_data.hdf5"
    tau_int_graph(N, dim, data_file, filename = "tau_int.pdf")

    pass