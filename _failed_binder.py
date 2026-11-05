


def tc_extraction(sources, filename = "tc_extraction.pdf"):
    '''
    Draws the plots of binder cumulants.
    '''

    plt.figure(figsize=(10, 6))

    for data_file in sources:

        print(f"Reading data from {data_file}...")
        
        base_name = os.path.basename(data_file)
        N = int(base_name.split('_')[3])
        dim = int(base_name.split('_')[1])

        with h5py.File(data_file, "r") as file:
            temperatures = np.array(file[f"dim_{dim}_N_{N}/temperatures"])
            filtered_data = np.array(file[f"dim_{dim}_N_{N}/filtered_data"])

        print(f"Done. Filtered data shape: {filtered_data.shape}")

        observables = np.array([[magnetization(model) for model in models_at_T] for models_at_T in filtered_data])

        binder_cumulants = np.array([binder_cumulant(magnetizations_at_T) for magnetizations_at_T in observables])

        plt.plot(
            temperatures,
            binder_cumulants,
            marker='.',
            linestyle='solid',
            label=f"N = {N}, dim = {dim}",
        )

    plt.xlabel('Temperature')
    plt.ylabel(r"Binder Cumulant $U_4$")
    plt.grid(True, which="both", ls="--")
    plt.title(f'Binder Cumulant vs Temperature')
    plt.legend()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()




@njit(cache = ALLOW_NUMBA_CACHING)
def binder_cumulant(M) -> float:
    '''
    Given the magnetization of an Ising model (at fixed T, N^d)
    compute the Binder cumulant.
    '''
    
    m2_mean = np.mean(np.power(M, 2))
    m4_mean = np.mean(np.power(M, 4))
    
    return 1.0 - (m4_mean / (3.0 * np.power(m2_mean, 2)))