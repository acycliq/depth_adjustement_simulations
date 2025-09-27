import pandas as pd
import numpy as np
import napari

from pointcloud import pointcloud, pointcloud_grid
from img import create_3d_label_image

def stage_data(single_cell_data, config):
    mcr = config['mcr']
    spacing_factor = config['spacing_factor']
    rng = config['rng']

    pc = pointcloud(single_cell_data, mcr, rng)
    pc, img_shape = pointcloud_grid(mcr, pc, spacing_factor, rng)
    label_image_3d = create_3d_label_image(img_shape, pc)

    # viewer = napari.Viewer(ndisplay=3)
    # # Add label image
    # labels_layer = viewer.add_labels(label_image_3d.transpose(2, 1, 0), name="Labeled Spheres", opacity=0.3)
    #
    # # Get the colors from the labels layer for each unique label
    # unique_labels = pc.label.unique()
    # color_map = {}
    # for label in unique_labels:
    #     color_map[label] = labels_layer.get_color(label)
    #
    # # Map colors to each point based on its label
    # point_colors = [color_map[label] for label in pc.label.values]
    #
    # # Add points layer with matching colors
    # viewer.add_points(pc[['x', 'y', 'z']].values,
    #                   properties={'label': pc.label},
    #                   name="Point Cloud",
    #                   face_color=point_colors,
    #                   size=1.5)
    #
    # napari.run()

    return pc, label_image_3d