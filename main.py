import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import pciSeq
from nb_utils import simulate_nb_matrices
from stage import stage_data, launch_napari
from plotting import confusion_matrix, plot_confusion_matrix


def main_loop(raw_counts, scRNAseq, cfg):

    # Randomly shuffle the columns of the simulated data
    if cfg['shuffle']:
        shuffled_indices = cfg['rng'].permutation(len(raw_counts.columns))
        raw_counts = raw_counts.iloc[:, shuffled_indices]

    pc, img3d = stage_data(raw_counts, cfg)
    # launch_napari(pc, img3d)

    spots = pc[['x', 'y', 'z', 'gene']]
    spots = spots.assign(score = 1)
    spots = spots.assign(intensity = 1)
    spots = spots.rename(columns={'z': 'z_plane',
                                  'gene': 'gene_name'})


    coo = [coo_matrix(d) for d in img3d]
    opts_3D = {
        'save_data': False,
        'launch_diagnostics': False,
        'launch_viewer': False,
        'Inefficiency': cfg['inefficiency'],
        'rGene': cfg['rGene'],
        'SpotReg': cfg['SpotReg'],
        'rSpot': cfg['rSpot'],
        'MisreadDensity': 1e-20,
        'nNeighbors': cfg['nNeighbors'],
        'InsideCellBonus':0,
        'CellCallTolerance': 0.05,
        'voxel_size':[1,1,1]
    }
    pciSeq.setup_logger()
    cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts_3D)

    mapping = dict(pc[['label','class']].drop_duplicates().values)
    cellData = cellData.assign(actual_class = cellData.Cell_Num.map(lambda d: mapping[d]))
    out = pd.DataFrame({
        'cell_label': cellData.Cell_Num.tolist(),
        'Estimated_class':  cellData.ClassName.values.tolist(),
        'Actual_class': cellData.actual_class.values.tolist(),
        'Prob': cellData.Prob.values.tolist(),
    })
    out['Actual==Best_class'] = out.Estimated_class == out.Actual_class

    classes = scRNAseq.columns.tolist()
    cm = confusion_matrix(classes, out)
    return out, cm, opts_3D

def get_scNAseq(name):
    if name == 'zeisel':
        # Get the Zeisel single cell data
        scRNAseq = pd.read_csv(os.path.join('data', 'scRNAseq', 'zeisel', 'l5_all_scRNAseq.csv'))
        scRNAseq = scRNAseq.set_index('Unnamed: 0')
    elif name == 'yao':
        # Get the Zeisel single cell data
        scRNAseq = pd.read_csv(os.path.join('data', 'scRNAseq', 'yao', 'scRNAseq_final.csv'))
        scRNAseq = scRNAseq.set_index('Unnamed: 0')
    else:
        raise ValueError('scRNAseq name not recognized')
    return scRNAseq



def app(cells_dict, cfg):
    use_replicates = cfg.get('use_replicates')
    count_multipliers = cfg.get('count_multipliers')
    if not isinstance(use_replicates, bool):
        raise ValueError("cfg['use_replicates'] must be either True or False")

    # Get the Zeisel single cell data
    scRNAseq = get_scNAseq(cfg['scRNAseq_name'])

    # 2. simulate single cell data
    my_cells = list(cells_dict.values())
    if use_replicates:
        # take the unique columns, then use `inv` to replicate them
        # so the final dataframe has one column per entry in my_cells,
        # with repeated columns where a cell class appears multiple times
        u_cells, inv = np.unique(my_cells, return_inverse=True)
        my_scRNAseq = scRNAseq[u_cells]
        sim_counts = simulate_nb_matrices(my_scRNAseq, cfg)# unique cols
        sim_counts = [d.iloc[:, inv] for d in sim_counts]  # expand back to original
    else:
        my_scRNAseq = scRNAseq[my_cells].copy()

        sim_counts = simulate_nb_matrices(my_scRNAseq, cfg)
        for df in sim_counts:
            for j in range(df.shape[1]):
                factor = count_multipliers[j] if j < len(count_multipliers) else 1
                col = df.iloc[:, j].to_numpy()        # get column as numpy array
                top_idx = col.argsort()[-25:]                   # indices of top 5 values
                col[top_idx] = col[top_idx] * factor           # scale those values
                df.iloc[:, j] = col                   # write back

    # set the confusion matrix
    cm = np.zeros([scRNAseq.shape[1], scRNAseq.shape[1]])

    #3 loop over the sampled single cell data and generate random spots
    num_iter = len(sim_counts)
    for i in range(num_iter):
        out, cm_temp, opts_3D = main_loop(sim_counts[i], scRNAseq, cfg)
        cm = cm + cm_temp

    # Combine config and opts_3D for complete settings display
    # opts_3D from the last iteration represents the final parameters used
    all_settings = {**cfg, **opts_3D}
    fig = plot_confusion_matrix(cm/num_iter, scRNAseq.columns.values, all_settings)
    fig.show(renderer="browser")

    return fig



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rng = np.random.default_rng(42)  # seed is set here

    config = {
        'n_samples': 100,  # Start with small test
        'use_replicates': False, # if true the counts and spatial coords are the same for cells with the same class
        'shuffle': False,  # if true then the order of the cells is shuffled
        'scRNAseq_name': 'zeisel',
        'count_multipliers': [1.0, 1.0, 1.0], # will scale the counts of the top 5 genes for cell1, cell2, cell3 by this factor
        'mcr': 18,
        'inefficiency': 1.0,
        'rGene': 20,
        'SpotReg': 1e-5, #regularization parameter, default 0.1
        'rSpot': 5,   # negative binomial spread, default 2
        'spacing_factor': 6, # regulates how close the pointclouds are to each other. If 2 then they are 2xCellRadius apart.
        'nNeighbors': 2,
        'rng': rng,
    }

    # pass here some cell and the corresponding class
    # my_cells = pd.read_csv(os.path.join('data', 'cells', 'zeisel', 'cells.csv'))
    # my_cells = my_cells.set_index("label")["class"].to_dict()
    my_cells = {
        1: 'DGGRC1',
        2: 'TEGLU24',
        3: 'DGGRC1',
        # 4: 'TEGLU21'
    }

    app(my_cells, config)