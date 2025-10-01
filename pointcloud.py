import pandas as pd
import numpy as np


def pointcloud(df, config):
    # generate random spots from the i-th simulated single cell data
    r = config['mcr']
    rng = config['rng']

    n_cols = df.shape[1]
    n_rows = df.shape[0]

    # Check for duplicate column names (same class cells)
    col_names = df.columns.tolist()
    has_duplicates = len(set(col_names)) < n_cols

    if has_duplicates:
        # Process cells one by one, reusing points for duplicate class names
        all_data = []
        class_points = {}  # Cache points for each class name

        for cell_idx in range(n_cols):
            class_name = col_names[cell_idx]
            cell_counts = df.iloc[:, cell_idx].values

            if class_name in class_points:
                # Reuse points from first occurrence of this class
                cell_data = class_points[class_name].copy()
                cell_data['label'] = cell_idx + 1
            else:
                # Generate new points for this class
                genes = df.index.tolist()
                gene_counts = cell_counts

                genes_expanded = np.repeat(genes, gene_counts.astype(int))
                counts = gene_counts.sum().astype(int)

                points = rng.normal(loc=[0, 0, 0], scale=r, size=(counts, 3))
                r_vals = r * np.ones(counts)

                cell_data = pd.DataFrame({
                    'gene': genes_expanded,
                    'x': points[:, 0],
                    'y': points[:, 1],
                    'z': points[:, 2],
                    'r': r_vals,
                    'label': cell_idx + 1,
                    'class': class_name
                })

                # Cache for future cells with same class
                class_points[class_name] = cell_data.copy()

            all_data.append(cell_data)

        data = pd.concat(all_data, ignore_index=True)
    else:
        # Original logic for unique classes
        genes = df.index.tolist()
        genes = np.tile(genes, [1, n_cols])

        actual_class = df.columns.tolist()
        actual_class = np.tile(actual_class, [n_rows, 1]).ravel(order='F')

        labels = 1 + np.arange(n_cols)
        labels = np.tile(labels, [n_rows, 1]).ravel(order='F')

        gene_counts = df.values.ravel(order='F')
        genes = np.repeat(genes, gene_counts)
        actual_class = np.repeat(actual_class, gene_counts)
        labels = np.repeat(labels, gene_counts)

        counts = gene_counts.sum()
        points = rng.normal(loc=[0, 0, 0], scale=r, size=(counts, 3))
        r_vals = r * np.ones(points.shape[0])

        data = np.column_stack((genes, points, r_vals, labels, actual_class))
        data = pd.DataFrame(data, columns=['gene', 'x', 'y', 'z', 'r', 'label', 'class'])

    data = data.astype({
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'r': np.float32,
        'label': np.int32
    })
    data = data.sort_values(by=['label', 'gene']).reset_index(drop=True)
    return data


def pointcloud_grid(radius, pointcloud, spacing_factor, rng, axis_mapping=None):
    """
    Arrange cells in 3D space using a regular grid on two axes and random positioning on the third axis.

    Parameters:
    -----------
    axis_mapping : dict, optional
        Maps grid and depth axes to coordinate systems.

        Keys:
        - 'grid_axis_0': First grid dimension (e.g., 'x')
        - 'grid_axis_1': Second grid dimension (e.g., 'y')
        - 'depth_axis': Random depth dimension (e.g., 'z')

        Examples:
        - Default (X-Y grid, Z depth):
          {'grid_axis_0': 'x', 'grid_axis_1': 'y', 'depth_axis': 'z'}
        - X-Z grid, Y depth:
          {'grid_axis_0': 'x', 'grid_axis_1': 'z', 'depth_axis': 'y'}
        - Y-Z grid, X depth:
          {'grid_axis_0': 'y', 'grid_axis_1': 'z', 'depth_axis': 'x'}
    """

    # Default axis mapping: X-Y grid, Z depth (preserves original behavior)
    if axis_mapping is None:
        axis_mapping = {'grid_axis_0': 'x', 'grid_axis_1': 'y', 'depth_axis': 'z'}

    radius = np.int32(radius)
    spacing_factor = np.int32(spacing_factor)

    class_names = np.unique(pointcloud['class'])
    n_cells = pointcloud.label.max()

    # Determine grid size - arrange cells in rectangular grid
    grid_dim_1 = int(np.sqrt(n_cells))  # Grid width (approximate square root)
    grid_dim_0 = int(np.ceil(n_cells / grid_dim_1))  # Grid height (ensure all cells fit)
    spacing = spacing_factor * radius  # Distance between cell centers

    # Create grid positions for the two grid axes
    axis_0_coords = np.arange(grid_dim_0) * spacing + spacing // 2  # Grid axis 0 positions
    axis_1_coords = np.arange(grid_dim_1) * spacing + spacing // 2  # Grid axis 1 positions

    # Generate all grid coordinate combinations and flatten to list
    axis_0_grid, axis_1_grid = np.meshgrid(axis_0_coords, axis_1_coords)  # Create 2D coordinate grids
    axis_1_centers = axis_1_grid.flatten()[:n_cells]  # Flatten and trim to exact number of cells
    axis_0_centers = axis_0_grid.flatten()[:n_cells]  # Flatten and trim to exact number of cells

    # Generate random positions for the depth axis
    depth_coords = rng.integers(2 * radius, 4 * radius + 1, size=n_cells)

    # Create label mapping for cell centroids
    labels = 1 + np.arange(pointcloud.label.max())

    # Map generic axis coordinates to specific x,y,z coordinates
    grid_axis_0 = axis_mapping['grid_axis_0']
    grid_axis_1 = axis_mapping['grid_axis_1']
    depth_axis = axis_mapping['depth_axis']

    # Create coordinate dictionaries for each axis
    coord_data = {'label': labels}
    coord_data[f'{grid_axis_0}c'] = axis_0_centers
    coord_data[f'{grid_axis_1}c'] = axis_1_centers
    coord_data[f'{depth_axis}c'] = depth_coords

    # Create dataframe with centroid coordinates for each cell
    centroids = pd.DataFrame(coord_data)

    # Merge centroids with pointcloud data to assign positions
    out = pointcloud.merge(centroids, on='label')

    # Shift all points from origin to their assigned positions
    for axis in ['x', 'y', 'z']:
        out[axis] = out[axis] + out[f'{axis}c']

    # Determine overall grid shape - map dimensions to correct axes
    shape_dict = {}
    shape_dict[grid_axis_0] = grid_dim_0 * spacing
    shape_dict[grid_axis_1] = grid_dim_1 * spacing
    shape_dict[depth_axis] = 6 * radius  # Sufficient depth

    # Return shape in (z, y, x) order for consistency with image conventions
    shape = (shape_dict['z'], shape_dict['y'], shape_dict['x'])

    return out, shape