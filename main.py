import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import pciSeq
from nb_utils import simulate_nb_matrices
from stage import stage_data
from plotting import confusion_matrix, plot_confusion_matrix


def main_loop(raw_counts, scRNAseq, cfg):

    # Randomly shuffle the columns of the simulated data
    shuffled_indices = cfg['rng'].permutation(len(raw_counts.columns))
    raw_counts = raw_counts.iloc[:, shuffled_indices]

    pc, img3d = stage_data(raw_counts, cfg)
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


def app(my_cells, cfg):

    # Get the Zeisel single cell data
    scRNAseq = pd.read_csv(os.path.join('data', 'l5_all_scRNAseq.csv'))
    scRNAseq = scRNAseq.set_index('Unnamed: 0')

    # 2. simulate single cell data
    my_scRNAseq = scRNAseq[list(my_cells.values())]
    sim_counts = simulate_nb_matrices(my_scRNAseq, cfg)

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
        'mcr': 18,
        'inefficiency': 1.0,
        'rGene': 20,
        'SpotReg': 0.01, #regularization parameter, default 0.1
        'rSpot': 5,   # negative binomial spread, default 2
        'spacing_factor': 4, # regulates how close the pointclouds are to each other. If 2 then they are 2xCellRadius apart.
        'nNeighbors': 6,
        'rng': rng,
    }

    my_cells = {
        1: 'ABC',
        2: 'ACTE1',
        3: 'ACTE2',
        4: 'DGGRC1',
        5: 'DGGRC2',
        6: 'DGNBL1',
        7: 'DGNBL2',
        8: 'EPEN',
        9: 'MFOL1',
        10: 'MFOL2',
        11: 'MGL1',
        12: 'MGL2',
        13: 'MGL3',
        14: 'MOL1',
        15: 'MOL2',
        16: 'MOL3',
        17: 'OPC',
        18: 'PER1',
        19: 'PER2',
        20: 'PER3',
        21: 'TEGLU1',
        22: 'TEGLU10',
        23: 'TEGLU11',
        24: 'TEGLU12',
        25: 'TEGLU13',
        26: 'TEGLU14',
        27: 'TEGLU2',
        28: 'TEGLU20',
        29: 'TEGLU21',
        30: 'TEGLU23',
        31: 'TEGLU24',
        32: 'TEGLU3',
        33: 'TEGLU4',
        34: 'TEGLU6',
        35: 'TEGLU7',
        36: 'TEGLU8',
        37: 'TEGLU9',
        38: 'TEINH10',
        39: 'TEINH11',
        40: 'TEINH12',
        41: 'TEINH13',
        42: 'TEINH14',
        43: 'TEINH15',
        44: 'TEINH16',
        45: 'TEINH17',
        46: 'TEINH18',
        47: 'TEINH19',
        48: 'TEINH20',
        49: 'TEINH21',
        50: 'TEINH4',
        51: 'TEINH5',
        52: 'TEINH6',
        53: 'TEINH7',
        54: 'TEINH8',
        55: 'TEINH9',
        56: 'VECA',
        57: 'VECC',
        58: 'VECV',
        59: 'VLMC1',
        60: 'VLMC2',
        61: 'VSMCA'
    }
    app(my_cells, config)