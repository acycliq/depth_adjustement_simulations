import pandas as pd
import numpy as np


def simulate_nb_matrices(expression_matrix, cfg):
    """
    Generate simulated gene count matrices assuming a negative binomial model.

    Parameters:
    - expression_matrix: pandas DataFrame with genes as rows and cell classes as columns.
      Each element is assumed to be the mean (μ) count.
    - num_simulations: number of simulated matrices to generate (default 10).
    - r: dispersion parameter of the negative binomial distribution (default 2).

    Returns:
    - A list of pandas DataFrames, each with the same shape and labels as the input.
    """
    r = cfg['mcr']
    num_samples = cfg['n_samples']
    rng = cfg['rng']
    inefficiency = cfg['inefficiency']
    rGene = cfg['rGene']

    # Convert the DataFrame values to a NumPy array for vectorized operations.
    mu_values = expression_matrix.values.astype(float)

    # Compute p for every element: p = r/(r + μ)
    p_values = r / (r + mu_values)

    nG = mu_values.shape[0]  # Number of genes
    simulated_matrices = []
    for i in range(num_samples):
        # Generate simulated counts for each element.
        # np.random.negative_binomial can work with array inputs for p.
        sim_counts = rng.negative_binomial(r, p_values)

        # apply the inefficiency
        if inefficiency == 1:
            print(' inefficiency == 1! I am not using etas and inefficiency')
        else:
            eta = rng.gamma(rGene, 1/rGene, nG)
            sim_counts = inefficiency * eta[:, None] * sim_counts

        # Wrap the simulated counts back into a DataFrame preserving row and column labels.
        sim_matrix = pd.DataFrame(sim_counts, index=expression_matrix.index, columns=expression_matrix.columns)
        simulated_matrices.append(sim_matrix)

    return simulated_matrices