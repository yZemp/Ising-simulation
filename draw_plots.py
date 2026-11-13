from datetime import timedelta
import os
import time
from datetime import timedelta
from typing import cast
from PIL.ImageQt import rgb
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from iminuit import Minuit, cost
from scipy.interpolate import CubicSpline
from iminuit import Minuit, cost

from hdf5_utils import read_data
from autocorrelation import tau_int_sokal, tau_int_blocking, autocorrelation
from mcmc_utils import metropolis_ising
from operators import energy, extract_tc_from_chi, magnetization, compute_chi
from analytical_curves import plot_analytical_curve


def graph_magnetization_convergence(sources, dim, filename = 'magnetization_convergence.png'):
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
        print(f"Plotting magnetization for source: {source}")

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
    with h5py.File(source, 'r') as f:
        temps = np.array(f[f"{list(f.keys())[0]}/temperatures"])
        t = np.linspace(temps[6], temps[-5], 100)
    plot_analytical_curve(dim = dim, t = t)

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

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize = (9, 9), sharex = True, gridspec_kw = {'height_ratios': [1, 1]})
    # fig, ax_down = plt.subplots(1, 1, figsize = (9, 5))
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

        if True:
            first = False

            tau_int = tau_int_blocking(energies, c = 20)[0]
            # ax_down.axvline(
            #     x = 20 * tau_int,
            #     color = current_color,
            #     linestyle = '--',
            #     linewidth = 1.5,
            #     zorder = 0,
            #     label = 'Thermalization time'
            # )
    
        step = int(.5 * tau_int) if tau_int > 0 else 1000
        ax_down.plot(
            energies[int(20 * tau_int):steps:step],
            range(0, len(energies[int(20 * tau_int):steps:step])),
            marker='o',
            color=mcolors.to_rgba(current_color, alpha=0.3),
            markerfacecolor=current_color,
            markeredgecolor=current_color,
            label=f'Initial state: {initial_state}'
        )

        burn_in = int(20 * tau_int)
        eq_energies = energies[burn_in:]

        # Normal distribution fit to the energy histogram
        # Not boltzmann, because g(E) * exp(-E/kT) for big lattices is a normal distribution
        def f_pdf(E, mu, sigma):
            return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((E - mu) / sigma)**2)

        c = cost.UnbinnedNLL(eq_energies, f_pdf)
        m = Minuit(c, mu = -200, sigma = 10)
        print(m.migrad())

        x = np.linspace(eq_energies.min(), eq_energies.max(), 100)
        ax_up.plot(
            x,
            f_pdf(x, mu = m.values['mu'], sigma = m.values['sigma']),
            color = mcolors.to_rgba(current_color, alpha = 1),
            linestyle = "dashed",
            linewidth = 2,
        )

        ax_up.hist(
            eq_energies,
            bins=(1 + int(np.log2(len(eq_energies)))),
            edgecolor=current_color,
            linewidth=1,
            linestyle='solid',
            facecolor=current_color,
            density=True,
            histtype='stepfilled',
            alpha=0.5,
            label=f'Initial state: {initial_state}'
        )

    ax_up.set_ylabel('Probability Density')
    ax_up.set_title(f'Energy Distribution: T = {T}, Shape = {models_samples.shape}')
    ax_up.legend()
    ax_up.grid()
    ax_up.set_axisbelow(True)
    
    ax_down.set_ylabel('Time')
    ax_down.get_yaxis().set_visible(False)
    ax_down.set_xlabel('Energy')
    ax_down.set_title(f'Energy Evolution: T = {T}, Shape = {models_samples.shape[1:]}')
    # ax_down.legend(loc='lower right')
    ax_down.legend()
    ax_down.grid()
    ax_down.set_axisbelow(True)
    

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1)

    plt.savefig(f"ED_{models_samples.shape[1:]}_{T}.pdf", format='pdf', bbox_inches = 'tight')
    plt.close()


def metropolis_ising_graphics_blocking(initial_state, T: float, steps: int, seed: int):
    '''
    NOTE:   This is (conceptually) a copy of the function above,
            but used to show the blocking method
    '''

    colors = [
        (253/255, 231/255, 37/255),  # yellow
        (53/255, 183/255, 121/255),  # green
        (72/255, 40/255, 120/255)    # purple
    ]

    LEN = 5_000

    print(f"Running Metropolis-Hastings for initial state: {initial_state.shape} at T = {T} for {steps} steps...")
    models_samples = metropolis_ising(initial_state, T, steps, seed)
    energies = np.array([energy(model) for model in models_samples])


    current_color = colors[1]

    tau_int, Nb_array, tau_int_estimates_array, vk_estimates_array, burn_in = tau_int_blocking(energies, c = 20)
    print(f"Tau_int_blocking: {tau_int}\nNb_array: {Nb_array}\nTau_int_estimates_array: {tau_int_estimates_array}\nBurn-in: {burn_in}")
    plateau_idx = np.where(tau_int_estimates_array == tau_int)[0][0]
    print("Plateau index:", plateau_idx)

    # BLOCKING
    n_plots = (plateau_idx + 1) // 2
    # n_plots = len(Nb_array)  # Show all blocking levels
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, n_plots))

    fig, axes = plt.subplots(n_plots, 1, figsize = (9, 2 * n_plots), sharex = True)
    if n_plots == 1: axes = [axes]

    original_data = energies[burn_in:]
    x_axis = np.arange(0, len(original_data), 1)

    block_size = 1
    data = original_data.copy()
    extended_data = np.repeat(data, block_size)

    for i in range(plateau_idx + 1):
        # Skip half plots
        if i % 2 == 0:
            block_size *= 2
            Nb = len(original_data) // block_size
            data = original_data[:Nb * block_size].reshape(Nb, block_size).mean(axis=1)
            extended_data = np.repeat(data, block_size)

            continue

        current_x = x_axis[:len(extended_data)]
        
        axes[i // 2].plot(
            current_x,
            extended_data,
            marker='None',
            color=mcolors.to_rgba(colors[i // 2], alpha=1.),
            linewidth=1.2,
            label=f'Block Size: {block_size}',
            drawstyle='steps-post'
        )

        axes[i // 2].set_ylabel('Energy')
        # axes[i // 2].set_xlabel('Time (steps)')
        axes[i // 2].legend(loc='lower right')
        axes[i // 2].grid()

        # Next iteration:
        block_size *= 2
        Nb = len(original_data) // block_size
        data = original_data[:Nb * block_size].reshape(Nb, block_size).mean(axis=1)
        extended_data = np.repeat(data, block_size)
        
    plt.suptitle(f'Energy Evolution: T = {T}, Shape = {models_samples.shape[1:]}')

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1)

    plt.savefig(f"ED_{models_samples.shape[1:]}_{T}.pdf", format='pdf', bbox_inches = 'tight')
    plt.close()


def autocorrelation_graph(N, dim, data_file = "tmp.hdf5", filename = "autocorrelation.pdf", T_index = 30):
    '''
    Plots the autocorrelation function of an observable O as a function of time (steps) 
    given the raw data stored in an HDF5 file.
    '''

    LEN = 2000

    # Not using read_data() here to economize memory usage
    with h5py.File(data_file, "r") as file:
        temperatures = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/temperatures"]))
        raw_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[T_index, :LEN])
        full_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[T_index])

    print(f"Raw data shape: {raw_data.shape} (T = {temperatures[T_index]:.2f})")

    times = np.arange(0, LEN, 1)
    acs = np.zeros_like(times, dtype = float)
    observables = np.array([magnetization(model) for model in raw_data])

    for i, t in enumerate(times):
        acs[i] = autocorrelation(t, observables)

    plt.figure(figsize=(10, 6))
    plt.plot(times, acs, label='Autocorrelation function', color=(53 / 255, 183 / 255, 121 / 255))
    plt.plot(0, acs[0], label=f"Initial value: {acs[0]:.2f}", marker='x', markersize=8, color=(72 / 255, 40 / 255, 120 / 255))


    # Showing Sokal's window
    full_observables = np.array([magnetization(model) for model in full_data])
    tau_int, M = tau_int_sokal(full_observables, c = 20)
    plt.axvline(x=M, color='red', linestyle='--', linewidth=2, 
            label=f'Window M={M} ($\\tau_{{int}}$={tau_int:.2f})')
    
    M_idx = int(M) + 1
    plt.fill_between(times[:M_idx], 0, acs[:M_idx], 
                     color=(53 / 255, 183 / 255, 121 / 255), alpha=0.5)


    plt.xlabel('Time (steps)')
    plt.ylabel('Autocorrelation')
    
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.title(f'Autocorrelation Function - T = {temperatures[T_index]:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"Autocorrelation graph saved to {filename}.")


def tau_int_graph(N, dim, data_file, filename = "tau_int.pdf", read_observables_from_file = False):
    '''
    Plots the integrated autocorrelation time (tau_int) with respect to magnetization
    as a function of temperature.
    '''

    csokal = 10.0
    cblocking = 10.0
    MAX_LENGTH = 1_000_000
    TESTING = False

    group_name = f"dim_{dim}_N_{N}"

    if read_observables_from_file:
        print("Reading observables from file...")
        observables_filename = f"observables_N_{N}_dim_{dim}.txt"
        with open(observables_filename, 'r') as f:
            header = f.readline()
            temperatures = np.array([float(x.split("=")[1]) for x in header.strip().split() if "T=" in x])
        observables = np.loadtxt(observables_filename, delimiter=" ")
    else:
        with h5py.File(data_file, "r") as file:
            raw_dataset = file[f"{group_name}/raw_data"]
            temperatures = np.array(cast(h5py.Dataset, file[f"{group_name}/temperatures"]))

            steps = min(raw_dataset.shape[1], MAX_LENGTH)
            
            observables = np.zeros((len(temperatures), steps), dtype=float)
            
            print("Computing observables (single temperature at a time)...")
            for i in range(len(temperatures)):
                print(f"Processing temperature {temperatures[i]:.2f} ({i + 1}/{len(temperatures)})...")
                print(f"Time is: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                if TESTING and i > 5: continue  # Testing
                # Slicing: loading only the necessary data for the current temperature to save memory
                data_T = raw_dataset[i, :steps, ...] 
                
                for step in range(steps):
                    observables[i, step] = magnetization(data_T[step, ...])
                # observables[i, :] = np.abs(np.mean(data_T, axis=(1, 2)))

            # Dump observables to a text file for further analysis
            observables_dump_filename = f"observables_N_{N}_dim_{dim}.txt"
            np.savetxt(observables_dump_filename, observables, delimiter=" ", header=" ".join([f"T={T:.2f}" for T in temperatures]))
            print(f"Observables dumped to {observables_dump_filename}")

    taus_sokal = np.zeros(len(temperatures), dtype=float)
    taus_sokal_errors = np.zeros(len(temperatures), dtype=float)
    taus_blocking = np.zeros(len(temperatures), dtype=float)
    taus_blocking_errors = np.zeros(len(temperatures), dtype=float)

    for i, T in enumerate(temperatures):
        if (i < 5 or i > 8) and TESTING: continue  # Testing
        print(f"Temperature: {T:.2f}")
        print(f"Time is: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        try:
            taus_sokal[i], M = tau_int_sokal(observables[i], c = csokal)
            taus_sokal_errors[i] = taus_sokal[i] * np.sqrt((4 * M + 2) / len(observables[i])) if M > 0 else np.nan
            taus_blocking[i], err, Nb_arr, _, _, _ = tau_int_blocking(observables[i], c = cblocking)
            taus_blocking_errors[i] = err
        except Exception as e:
            print(f"Warning: failed at T = {T:.2f}: {type(e).__name__}: {e}")
            taus_sokal[i] = np.nan
            taus_sokal_errors[i] = np.nan
            taus_blocking[i] = np.nan
            taus_blocking_errors[i] = np.nan

    # fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize = (9, 9), sharex = True, gridspec_kw = {'height_ratios': [1, 1]})
    fig, (ax_up) = plt.subplots(1, 1, figsize = (9, 9))
    
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
        temperatures + 0.01, # small offset for better visibility
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
    ax_up.set_yscale('log')
    ax_up.grid(True, which="both", ls="--")
    ax_up.set_title(r"$\tau_{int}$" + f" - N = {N}, dim = {dim}, c_sokal = {csokal}, c_blocking = {cblocking}")
    ax_up.legend()

    # ax_down.plot(
    #     temperatures,
    #     abs(taus_sokal - taus_blocking),
    #     marker = ".",
    #     label = r"$\tau_{int} difference$",
    #     linestyle = "solid",
    #     color = 'black'
    #     )
    # ax_down.set_xlabel('Temperature')
    # ax_down.set_ylabel(r"Difference")
    # ax_down.grid(True, which="both", ls="--")
    # ax_down.legend()

    plt.savefig(filename, format='pdf', bbox_inches = 'tight')
    plt.close()


def tau_blocking_plateau(N, dim, data_file, filename = "tau_blocking_plateau.pdf", T_index = 30):
    '''
    Plots the blocking method for estimating tau_int, showing the plateau.
    '''

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/temperatures"]))
        raw_data = np.array(cast(h5py.Dataset, file[f"dim_{dim}_N_{N}/raw_data"])[T_index])

    print(f"Raw data shape: {raw_data.shape}")

    observables = np.array([magnetization(model) for model in raw_data])

    tau_int, err, Nb_array, tau_int_estimates_array, _, _ = tau_int_blocking(observables, c = 10.0)

    k = len(observables) / Nb_array[np.where(tau_int_estimates_array == tau_int)[0][0]]  # Number of samples per block at the plateau
    tau_int, M = tau_int_sokal(observables, c = 5.0)
    err_array = tau_int_estimates_array * np.sqrt(2.0 / (Nb_array - 1))
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(Nb_array, tau_int_estimates_array, yerr=err_array, marker='o', capsize=3, ecolor='#482878', color='#482878', label=r"$\tau_{int}$ estimates")
    plt.axhline(y=tau_int, color="#35B779", linestyle='solid', linewidth=1.5, label=f'Plateau value: {tau_int:.2f}')
    plt.xlabel('Number of blocks (Nb)')
    plt.ylabel(r"$\tau_{int}$ estimates")
    plt.xscale('log')
    plt.title(f'Blocking Method - N = {N}, dim = {dim}, T = {temperatures[T_index]:.2f}')
    plt.plot([], [], ' ', label=f"k = {k:.2f} samples per block at plateau")
    plt.plot([], [], ' ', label=f"M = {M:.2f}")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

    # Dump data to a text file for further analysis
    data_dump_filename = f"tau_blocking_plateau_data_N_{N}_dim_{dim}_T_{temperatures[T_index]:.2f}.txt"
    np.savetxt(data_dump_filename, np.column_stack((Nb_array, tau_int_estimates_array)), header="Nb\tTau_int_estimates", comments='')
    print(f"Data dumped to {data_dump_filename}")


def chi_graph(N, dim, data_file, filename = "extracting_tc.pdf"):
    '''
    Interpolates and plots chi(T),
    extracts the critical temperature T_C from the minimum of the derivative of chi(T),
    and saves the image to a PDF file.
    '''

    print(f"Reading data from {data_file}...")

    with h5py.File(data_file, "r") as file:
        temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
        chi_values = np.array(file[f"dim_{dim}_N_{N}/chi_values"])
        chi_errors = np.array(file[f"dim_{dim}_N_{N}/chi_errors"])

    print("Done.")

    # Selecting points around the maximum of chi for interpolation 
    points_before = 1
    points_after = 1

    max_chi_index = np.argmax(chi_values)
    interp_temps = temperatures[max_chi_index - points_before:max_chi_index + points_after + 1]
    chi_interpolation = CubicSpline(interp_temps, chi_values[max_chi_index - points_before:max_chi_index + points_after + 1])
    chi_derivative = chi_interpolation.derivative()

    # Find the maximum of chi(T) by finding the roots of the derivative
    all_roots = chi_derivative.roots()
    
    # Filter roots to be within the interpolation range
    t_min, t_max = interp_temps[0], interp_temps[-1]
    valid_roots = [r for r in all_roots if t_min <= r <= t_max]
    
    t_max_discrete = temperatures[max_chi_index]
    
    if valid_roots:
        # Select the root closest to the discrete maximum temperature
        tc = min(valid_roots, key=lambda r: abs(r - t_max_discrete))
    else:
        print("Warning: No valid roots found within the interpolation range. Using discrete maximum temperature.")
        tc = t_max_discrete

    chi_second_derivative = chi_interpolation.derivative(nu=2)
    curvature = abs(chi_second_derivative(tc))
    err_chi_max = chi_errors[max_chi_index]
    tc_error = np.sqrt(2 * err_chi_max / curvature) if curvature > 0 else 0.0
    
    # Max error is half the temperature discretization step
    dT = temperatures[max_chi_index] - temperatures[max_chi_index - 1]
    tc_error = max(tc_error, dT / 2)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: chi
    color1 = "#482878"
    ax1.set_xlabel('Temperature (T)')
    ax1.set_ylabel(r'Magnetic Susceptibility ($\chi$)')
    ax1.errorbar(
        temperatures,
        chi_values,
        yerr=chi_errors,
        marker='o',
        markersize=4,
        linestyle='solid',
        capsize = 3,
        linewidth=.5,
        color=color1,
        alpha=0.5,
        label=r'$\chi$ data'
    )
    ax1.errorbar(
        temperatures[max_chi_index - points_before:max_chi_index + points_after + 1],
        chi_values[max_chi_index - points_before:max_chi_index + points_after + 1],
        yerr=chi_errors[max_chi_index - points_before:max_chi_index + points_after + 1],
        marker='o',
        markersize=6,
        linestyle='None',
        capsize = 3,
        color=color1,
        label=r'$\chi$ data used for interpolation'
    )

    # Plot the interpolation of chi
    fine_temps = np.linspace(interp_temps[0], interp_temps[-1], 500)
    ax1.plot(fine_temps, chi_interpolation(fine_temps), linestyle='solid', color=color1, label=r'$\chi$ interpolation')

    color2 = "#35B779"

    ax1.axvline(x=tc, color=color2, alpha=1, linestyle='--', linewidth=1.5)
    ax1.axvspan(x=tc, xmin=tc - tc_error, xmax=tc + tc_error, color=color2, alpha=.4, linestyle='none', label=f"Max $\\chi$ at $T_c = {tc:.3f} \\pm {tc_error:.3f}$")
   
    # Accepted critical temperature lines for reference
    accepted_Tc = 2.269 if dim == 2 else (4.5115 if dim == 3 else -1.0)
    if accepted_Tc > 0:
        ax1.axvline(x=accepted_Tc, color="#FDE725", linestyle='--', linewidth=2, label=f'Accepted $T_c \\approx {accepted_Tc:.3f}$')

    if dim == 2:
        xmin = 2.0
        xmax = 2.6
    elif dim == 3:
        xmin = 4.0
        xmax = 5.0

    ax1.set_xlim(xmin, xmax)

    plt.title(f'$T_C$ extraction - N = {N}, dim = {dim}')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()


def true_tc_extraction(sources, dim, filename = "true_tc_extraction.pdf"):
    '''
    Given a list of HDF5 files with magnetization data for different Ns, 
    computes T_C(N): the estimated critical temperature for each N.
    Then, it plots the results for each N,
    and fits the curve T_C(N) = T_C(inf) + a * N^(-1/nu) to extract the true critical temperature T_C(inf).

    with:
        nu = 1 for dim = 2
        nu = 0.63 for dim = 3
    '''

    if len(sources) <= 0:
        raise ValueError('`sources` must contain at least one file path')
    
    if dim == 2:
        nu = 1.0
    elif dim == 3:
        nu = 0.63
    else:
        raise ValueError("Dimension must be 2 or 3 for T_C extraction.")

    # Sorting by N
    def get_n_from_file(filepath):
        with h5py.File(filepath, 'r') as f:
            group_name = list(f.keys())[0]
            return int(group_name.split('_')[3])

    sources = sorted(sources, key = get_n_from_file)

    Tc_values = []
    Tc_errors = []
    Ns = []

    for i, source in enumerate(sources):
        print(f"Reading data from {source}...")

        N = source.split('_')[6] # Extract N from filename
        print(N)
        with h5py.File(source, "r") as file:
            temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
            chi_values = np.array(file[f"dim_{dim}_N_{N}/chi_values"])
            chi_errors = np.array(file[f"dim_{dim}_N_{N}/chi_errors"])

        print("Done.")

        print(f"Computing T_C(N) for N = {N} from source: {source} ({i + 1}/{len(sources)})")

        tc, tc_error = extract_tc_from_chi(chi_values, chi_errors, temperatures, points_before = 1, points_after = 1)
        Tc_values.append(tc)
        Tc_errors.append(tc_error)
        Ns.append(int(N))

    print(f"Final T_C(N) values: {Tc_values}")
    print(f"Final T_C(N) errors: {Tc_errors}")
    print(f"Final Ns: {Ns}")

    plt.figure(figsize=(10, 6))

    # Fit the curve T_C(N) = T_C(inf) + a * N^(-1/nu)
    def fit_function(N, Tc_inf, a):
        return Tc_inf + a * N**(-1/nu)

    Ns = np.array(Ns)
    Tc_values = np.array(Tc_values)
    Tc_errors = np.array(Tc_errors)

    least_squares = cost.LeastSquares(Ns, Tc_values, Tc_errors, fit_function)
    m = Minuit(least_squares, Tc_inf = 2.0, a = 1.0)
    m.migrad()
    m.hesse()

    x_data = Ns**(-1/nu)
    plt.errorbar(
        x_data,
        Tc_values,
        yerr=Tc_errors, 
        marker='o',
        linestyle='solid',
        alpha=0.7,
        color="#482878",
        # color="#FDE725",
        capsize=3,
        label=r'Estimated $T_C(N)$'
    )

    N_dense = np.logspace(np.log10(min(Ns)), np.log10(max(Ns)*1000), 200)
    x_fit = N_dense**(-1/nu)
    y_fit = fit_function(N_dense, m.values['Tc_inf'], m.values['a'])
    plt.plot(
        x_fit,
        y_fit,
        linestyle='solid',
        color="#35B779",
        linewidth=2,
        label=f'Fit: $T_C(N) = T_C(\\infty) + a \\cdot N^{{-1/{nu}}}$',
        zorder=100
    )

    tc_inf_val = m.values['Tc_inf']
    tc_inf_err = m.errors['Tc_inf']
    plt.errorbar(
        x_fit[-1],
        tc_inf_val,
        tc_inf_err,
        marker='o',
        color="#FDE725", 
        markersize=8,
        linestyle='None',
        capsize=5,
        label=f'Extrapolated $T_C(\\infty) \\approx {tc_inf_val:.3f} \\pm {tc_inf_err:.3f}$',
        zorder=101
    )

    # plt.axvline(0, color='grey', linestyle=':', alpha=0.5)
    plt.xlabel(rf'$N^{{-1/{nu}}}$')
    plt.ylabel(r'$T_C(N)$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()




##################################################################################################
# Main execution block
##################################################################################################

if __name__ == '__main__':
    start = time.perf_counter()
    print("Starting draw_plots.py...")
    print(f"The time is: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    

    #########################################################################################
    #########################################################################################
    # MAGNETIZATION VISUALIZATION EXAMPLE USAGE

    # Ns = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 500]
    # dim = 1

    # Ns = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    # Ns = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    # dim = 2
    
    # Ns = [4, 6, 8, 10, 15, 30]
    # dim = 3

    # sources = [r"E:\simulations_data\sweep_based_chains\dim_{dim}_N_{N}_data.hdf5".format(dim = dim, N = N) for N in Ns]
    # graph_magnetization_convergence(sources, dim, filename = f"magnetization_convergence_{dim}D.pdf")


    #########################################################################################
    # Single file example usage
    # sources = [r"dim_2_N_50_data.hdf5"]
    # graph_magnetization_convergence(sources, filename = f"tmp.png")



    #########################################################################################
    #########################################################################################
    # MARKOV CHAIN VISUALIZATION EXAMPLE USAGE

    # N = (15, 15)
    # SEED = 69
    # np.random.seed(SEED)

    # x1 = np.random.choice([-1, 1], size = N)
    # x2 = np.ones(N, dtype = np.int8)
    # x3 = - np.ones(N, dtype = np.int8)

    # starting_configs = {
    #     "random": x1,
    #     "all_up": x2,
    #     "all_down": x3
    # }

    # metropolis_ising_graphics(starting_configs, T = 2.26, steps = 200_000, seed = SEED)



    #########################################################################################
    #########################################################################################
    # Markov Chain WITH BLOCKING VISUALIZATION EXAMPLE USAGE

    # N = (15, 15)
    # SEED = 420
    # np.random.seed(SEED)

    # starting_config = np.random.choice([-1, 1], size = N)

    # metropolis_ising_graphics_blocking(starting_config, T = 5.26, steps = 200_000, seed = SEED)




    #########################################################################################
    #########################################################################################
    # AUTOCORRELATION VISUALIZATION EXAMPLE USAGE

    # N = 100
    # dim = 1
    # data_file = "E:\\simulations_data\\" + f"dim_{dim}_N_{N}" + "_data.hdf5"
    # data_file = f"simulations_data/dim_{dim}_N_{N}" + "_data.hdf5"
    # autocorrelation_graph(N, dim, data_file, filename = "autocorrelation_with_window.pdf", T_index = 30) 




    #########################################################################################
    #########################################################################################
    # Tau_int VISUALIZATION EXAMPLE USAGE

    # N = 500
    # dim = 1
    # data_file = f"E:\\simulations_data\\sweep_based_chains\\dim_{dim}_N_{N}" + "_data.hdf5"
    # tau_int_graph(N, dim, data_file, filename = f"tau_int__sweeps_better_N_{N}_dim_{dim}.pdf", read_observables_from_file = False)




    #########################################################################################
    #########################################################################################
    # Blocking plateau VISUALIZATION EXAMPLE USAGE

    N = 25
    dim = 2
    data_file = f"E:\\simulations_data\\dim_{dim}_N_{N}" + "_data.hdf5"
    tau_blocking_plateau(N, dim, data_file, filename = f"tau_blocking_plateau_N_{N}_dim_{dim}.pdf", T_index = 50)


    #########################################################################################
    #########################################################################################
    # Simple T_C extraction VISUALIZATION EXAMPLE USAGE

    # N = 100
    # dim = 2
    # N = 30
    # dim = 3
    # data_file = f"E:\\simulations_data\\sweep_based_chains\\dim_{dim}_N_{N}" + "_data.hdf5"
    # chi_graph(N, dim, data_file, filename = f"tc_extraction_{N}_{dim}.pdf")


    #########################################################################################
    #########################################################################################
    # TRUE T_C extraction VISUALIZATION EXAMPLE USAGE

    # Ns = [5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 500]
    # dim = 1

    # Ns = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
    # Ns = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    # dim = 2
    
    # Ns = [4, 6, 8, 10, 15, 30]
    # dim = 3

    # sources = [r"E:\simulations_data\sweep_based_chains\dim_{dim}_N_{N}_data.hdf5".format(dim = dim, N = N) for N in Ns]
    # true_tc_extraction(sources, dim, filename = f"true_tc_extraction_{dim}D.pdf")
    
    end = time.perf_counter()
    print(f"Time elapsed since draw_plots.py was run = {timedelta(seconds = end - start)}")