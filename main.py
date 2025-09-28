import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import pciSeq
from nb_utils import simulate_nb_matrices
from stage import stage_data
from plotting import confusion_matrix, plot_confusion_matrix


def main_loop(scRNAseq_temp, cfg):

    # Randomly shuffle the columns of the simulated data
    shuffled_columns = cfg['rng'].permutation(scRNAseq_temp.columns)
    scRNAseq_temp = scRNAseq_temp[shuffled_columns]

    pc, img3d = stage_data(scRNAseq_temp, cfg)
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
        'nNeighbors': 6,
        'InsideCellBonus':0,
        'CellCallTolerance': 0.05,
        'voxel_size':[1,1,1]
    }
    pciSeq.setup_logger()
    cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq_temp, opts=opts_3D)

    mapping = dict(pc[['label','class']].drop_duplicates().values)
    cellData = cellData.assign(actual_class = cellData.Cell_Num.map(lambda d: mapping[d]))
    out = pd.DataFrame({
        'cell_label': cellData.Cell_Num.tolist(),
        'Estimated_class':  cellData.ClassName.values.tolist(),
        'Actual_class': cellData.actual_class.values.tolist(),
        'Prob': cellData.Prob.values.tolist(),
    })
    out['Actual==Best_class'] = out.Estimated_class == out.Actual_class

    classes = scRNAseq_temp.columns.tolist()
    cm = confusion_matrix(classes, out)
    return out, cm, opts_3D


def app(cfg):

    # Get the Zeisel single cell data
    scRNAseq = pd.read_csv(os.path.join('data', 'l5_all_scRNAseq.csv'))
    scRNAseq = scRNAseq.set_index('Unnamed: 0')

    # 2. simulate single cell data
    sim_scRNAseq = simulate_nb_matrices(scRNAseq, cfg)

    # set the confusion matrix
    cm = np.zeros([scRNAseq.shape[1], scRNAseq.shape[1]])

    #3 loop over the sampled single cell data and generate random spots
    num_iter = len(sim_scRNAseq)
    for i in range(num_iter):
        out, cm_temp, opts_3D = main_loop(sim_scRNAseq[i], cfg)
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
        'n_samples': 1000,
        'mcr': 18,
        'inefficiency': 1.0,
        'rGene': 20,
        'SpotReg': 0.01, #regularization parameter, default 0.1
        'rSpot': 5,   # negative binomial spread, default 2
        'spacing_factor': 2,
        'rng': rng,
    }
    app(config)


