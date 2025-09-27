import numpy as np
# import napari
import pandas as pd


def create_3d_label_image(shape, pointclouds):
    """
    Creates a 3D label image with labeled spheres.

    Parameters:
        shape (tuple): Output image shape (Z, Y, X).
        spheres (list of tuples): Each tuple is (z, y, x, radius, label_id).

    Returns:
        np.ndarray: 3D array with labeled spheres.
    """
    label_img = np.zeros(shape, dtype=np.uint16)

    for label in np.unique(pointclouds.label):
        pointcloud = pointclouds[pointclouds.label == label]
        centroid = pointcloud[['xc', 'yc', 'zc', 'r']].drop_duplicates()
        assert centroid.shape[0] == 1, "centroid should be unique"
        xc, yc, zc, r = centroid.values[0]
        xc = xc.astype(np.int32)
        yc = yc.astype(np.int32)
        zc = zc.astype(np.int32)
        r = r.astype(np.int32)

        # label = np.int32(pointcloud.label)
        # Define the bounding box for the sphere, clamping to image boundaries.
        z_min = max(zc - r, 0)
        z_max = min(zc + r + 1, shape[0])
        y_min = max(yc - r, 0)
        y_max = min(yc + r + 1, shape[1])
        x_min = max(xc - r, 0)
        x_max = min(xc + r + 1, shape[2])

        # Create coordinate arrays for the bounding box using ogrid.
        zz, yy, xx = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

        # Calculate the sphere mask within the bounding box.
        mask = ((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2) <= r ** 2

        # Update only the region of the sphere.
        sub_region = label_img[z_min:z_max, y_min:y_max, x_min:x_max]
        sub_region[mask] = label
        label_img[z_min:z_max, y_min:y_max, x_min:x_max] = sub_region


    return label_img


