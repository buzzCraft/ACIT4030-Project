import numpy as np
import plotly.graph_objects as go


def visualize_rotate(data):
    """
    Creates a 3D plot with an animated rotation effect around the Z-axis.
    Parameters:
        data (list of plotly.graph_objs): A list of Plotly graph objects (e.g., Mesh3d or Scatter3d).
    Returns:
        plotly.graph_objs.Figure: A Plotly figure object with the rotation animation set up.
    """
    # Initial camera eye position
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames = []

    # Define a rotation function around the Z-axis
    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    # Create frames for the animation
    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(
            dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze)))))
        )

    # Create the figure with the rotation animation
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor="left",
                    yanchor="bottom",
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=50, redraw=True),
                                    transition=dict(duration=0),
                                    fromcurrent=True,
                                    mode="immediate",
                                ),
                            ],
                        )
                    ],
                )
            ]
        ),
        frames=frames,
    )

    return fig


def show_mesh(x, y, z, i, j, k):
    """
    Visualizes a 3D mesh and applies a rotation animation.
    Parameters:
        x, y, z (array-like): Coordinates of the mesh vertices.
        i, j, k (array-like): Indices of the vertices that form the mesh's triangles.
    """
    visualize_rotate(
        [go.Mesh3d(x=x, y=y, z=z, color="lightpink", opacity=0.50, i=i, j=j, k=k)]
    ).show()


def show_scatter(x, y, z):
    """
    Visualizes a 3D scatter plot and applies a rotation animation.
    Parameters:
        x, y, z (array-like): Coordinates of the scatter plot points.
    """
    visualize_rotate([go.Scatter3d(x=x, y=y, z=z, mode="markers")]).show()
