import pandas as pd
import numpy as np


def pointcloud(df, r, rng):
    # generate random spots from the i-th simulated single cell data
    # r = 9
    n_cols = df.shape[1]
    n_rows = df.shape[0]

    genes = df.index.tolist()
    genes = np.tile(genes,[1, n_cols])

    actual_class = df.columns.tolist()
    actual_class = np.tile(actual_class, [n_rows,1]).ravel(order='F')

    labels = 1 + np.arange(n_cols)
    labels = np.tile(labels, [n_rows,1]).ravel(order='F')

    gene_counts = df.values.ravel(order='F')
    genes = np.repeat(genes, gene_counts)
    actual_class = np.repeat(actual_class, gene_counts)
    labels = np.repeat(labels, gene_counts)

    counts = gene_counts.sum()

    points = rng.normal(loc=[0, 0, 0], scale=r, size=(counts, 3))
    r = r * np.ones(points.shape[0])

    data = np.column_stack((genes, points, r, labels, actual_class))
    data = pd.DataFrame(data, columns=['gene', 'x', 'y', 'z', 'r', 'label', 'class']).astype({
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'r': np.float32,
        'label': np.int32
    })
    data = data.sort_values(by=['label', 'gene']).reset_index(drop=True)
    return data


def pointcloud_grid(radius, pointcloud, spacing_factor, rng):

    radius = np.int32(radius)
    spacing_factor = np.int32(spacing_factor)

    class_names = np.unique(pointcloud['class'])
    n_cells = pointcloud.label.max()

    # Determine grid size
    grid_x = int(np.sqrt(n_cells))
    grid_z = int(np.ceil(n_cells / grid_x))
    spacing = spacing_factor * radius

    # Create grid positions for y and x
    x_coords = np.arange(grid_x) * spacing + spacing // 2
    z_coords = np.arange(grid_z) * spacing + spacing // 2

    # Create a grid of x and y positions and flatten them
    zc, xc = np.meshgrid(z_coords, x_coords)  # X and Y will have shape (grid_y, grid_x)
    zc = zc.flatten()[:n_cells]
    xc = xc.flatten()[:n_cells]

    # Generate random z positions for all spheres at once
    y_coords = rng.integers(2 * radius, 4 * radius + 1, size=n_cells)

    # Create label ids (starting at 1)
    labels = 1 + np.arange(pointcloud.label.max())

    centroids = pd.DataFrame({'label': labels,'zc': zc, 'xc': xc, 'yc': y_coords})

    # assign a centroid to each pointcloud
    out = pointcloud.merge(centroids, on='label')

    # shift the coordinates
    out.x = out.x + out.xc
    out.y = out.y + out.yc
    out.z = out.z + out.zc

    # Determine overall grid shape (depth, height, width)
    max_x = grid_x * spacing
    max_z = grid_z * spacing
    max_y = 6 * radius  # Sufficient depth
    shape = (max_z, max_y, max_x)


    return out, shape