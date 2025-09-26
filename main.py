import os
import pandas as pd
import numpy as np

from nb_utils import simulate_nb_matrices

def app():
    # Get the Zeisel single cell data
    scRNAseq = pd.read_csv(os.path.join('data', 'l5_all_scRNAseq.csv'))
    scRNAseq = scRNAseq.set_index('Unnamed: 0')

    rng = np.random.default_rng(42)  # seed is set here

    # 2. simulate single cell data
    sim_dfs = simulate_nb_matrices(scRNAseq, rng, r=2.0, inefficiency=1.0, rGene=20)
    sim_df = sim_dfs[0]

    #3 loop over the sampled single cell data and generate random spots
    for i in range(len(sim_dfs)):
        # generate random spots from the i-th simulated single cell data
        zc, yc, xc = [0, 0, 0]
        r = 9
        df = sim_dfs[i]
        df = df[['ABC', 'ACTE1']]
        n_cols = df.shape[1]
        n_rows = df.shape[0]

        genes = df.index.tolist(),
        genes = np.tile(genes,[1, n_cols])

        actual_class = df.columns.tolist()
        actual_class = np.tile(actual_class, [n_rows,1]).ravel(order='F')

        labels = df.columns.tolist()
        labels = np.tile(labels, [n_rows,1]).ravel(order='F')

        gene_counts = df.values.ravel(order='F')
        genes = np.repeat(genes, gene_counts)
        actual_class = np.repeat(actual_class, gene_counts)
        labels = np.repeat(labels, gene_counts)

        counts = gene_counts.sum()

        points = rng.normal(loc=[zc, yc, xc], scale=r, size=(counts, 3))

        out = np.column_stack((genes, points, labels, actual_class))
        out = pd.DataFrame(out, columns=['gene', 'x', 'y', 'z', 'label', 'class'])
        out = out.sort_values(by=['label', 'gene']).reset_index(drop=True)
        print('done')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app()


