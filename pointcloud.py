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

    # Determine grid size - arrange cells in rectangular grid
    grid_y = int(np.sqrt(n_cells))  # Grid width (approximate square root)
    grid_x = int(np.ceil(n_cells / grid_y))  # Grid height (ensure all cells fit)
    spacing = spacing_factor * radius  # Distance between cell centers

    # Create grid positions for X and Y coordinates
    x_coords = np.arange(grid_x) * spacing + spacing // 2  # X positions with offset
    y_coords = np.arange(grid_y) * spacing + spacing // 2  # Y positions with offset

    # Generate all grid coordinate combinations and flatten to list
    xc, yc = np.meshgrid(x_coords, y_coords)  # Create 2D coordinate grids
    yc = yc.flatten()[:n_cells]  # Flatten and trim to the exact number of cells
    xc = xc.flatten()[:n_cells]  # Flatten and trim to the exact number of cells

    # Generate random Z positions (depth) for each cell
    z_coords = rng.integers(2 * radius, 4 * radius + 1, size=n_cells)

    # Create label mapping for cell centroids
    labels = 1 + np.arange(pointcloud.label.max())

    # Create dataframe with centroid coordinates for each cell
    centroids = pd.DataFrame({'label': labels,'yc': yc, 'xc': xc, 'zc': z_coords})

    # Merge centroids with pointcloud data to assign positions
    out = pointcloud.merge(centroids, on='label')

    # Shift all points from origin to their assigned grid positions
    out.x = out.x + out.xc  # Move to X grid position
    out.y = out.y + out.yc  # Move to Y grid position
    out.z = out.z + out.zc  # Add random depth variation

    # Determine overall grid shape (depth, height, width)
    max_x = grid_x * spacing
    max_y = grid_y * spacing
    max_z = 6 * radius  # Sufficient depth
    shape = (max_z, max_y, max_x)


    return out, shape