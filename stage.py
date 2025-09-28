import pandas as pd
import numpy as np
import napari

from pointcloud import pointcloud, pointcloud_grid
from img import create_3d_label_image

def stage_data(raw_counts, config):
    mcr = config['mcr']
    spacing_factor = config['spacing_factor']
    rng = config['rng']

    pc = pointcloud(raw_counts, mcr, rng)
    pc, img_shape = pointcloud_grid(mcr, pc, spacing_factor, rng)
    label_image_3d = create_3d_label_image(img_shape, pc)

    # launch_napari(pc, label_image_3d)

    return pc, label_image_3d


def launch_napari(pc, label_image_3d):
    viewer = napari.Viewer(ndisplay=3)
    # Make axes obvious: show labels and standard RGB coloring (X=red, Y=green, Z=blue)
    viewer.axes.visible = True
    try:
        viewer.axes.labels = True
        viewer.axes.colored = True
    except Exception:
        pass
    # Label dimensions explicitly in order (Z, Y, X)
    try:
        viewer.dims.axis_labels = ("Z", "Y", "X")
        # Force axis colors to conventional mapping: Z=blue, Y=green, X=red
        # Many napari themes default to a CYM cycle, which can appear as X:cyan, Y:magenta, Z:yellow.
        # viewer.dims.axis_colors = ("blue", "green", "red")
    except Exception:
        pass
    # Add label image in (Z, Y, X) so left-right=X and bottom-top=Y
    labels_layer = viewer.add_labels(label_image_3d, name="Labeled Spheres", opacity=0.3)

    # Get the colors from the labels layer for each unique label
    unique_labels = pc.label.unique()
    color_map = {}
    for label in unique_labels:
        color_map[label] = labels_layer.get_color(label)

    # Map colors to each point based on its label
    point_colors = [color_map[label] for label in pc.label.values]

    # Add points layer with matching colors (coords as Z, Y, X to match labels)
    viewer.add_points(pc[['z', 'y', 'x']].values,
                      properties={'label': pc.label},
                      name="Point Cloud",
                      face_color=point_colors,
                      size=1.5)

    # Show scale bar for orientation/size
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'px'

    try:
        viewer.camera.set_view_direction(
            view_direction=(0, -1, 0),  # Looking along Y-axis
            up_direction=(1, 0, 0)    # Z-axis pointing up
        )
    except Exception:
        pass

    napari.run()
