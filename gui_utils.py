import io
import os
import random
import string
import tempfile
import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import plotly.graph_objects as go

def make_csv_infer(stl_file, foot):
    df = pd.DataFrame(
        data=[
            {
                "No.": "No Need",
                "Foot": "L" if foot.lower() == "left" else "R",
                "3D": stl_file,
            }
        ]
    )

    temp_csv_path = random_file_path("csv")
    df.to_csv(temp_csv_path, index=False)
    return temp_csv_path


def random_file_path(file_ext):
    # Create a temporary directory if it doesn't exist
    temp_dir = tempfile.mkdtemp(prefix="temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Generate a random filename
    random_filename = "".join(
        random.choices(string.ascii_letters + string.digits, k=10)
    )
    file_path = os.path.join(temp_dir, f"{random_filename}.{file_ext}")

    return file_path


def render_3d(file):
    # Check if the file is None
    if file is None:
        return None
    
    file_name = getattr(file, "name", file)

    if file_name.lower().endswith('.stl'):
        # Load STL file using the provided file name
        mesh = o3d.io.read_triangle_mesh(file_name)

        # Check if the loaded mesh is empty
        if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            return None

        # Convert vertices and triangles to NumPy arrays
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Create a Plotly figure
        fig = go.Figure()

        # Add the 3D mesh to the figure
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.5,
            )
        )
    elif file_name.lower().endswith('.txt'):
        points = np.loadtxt(file_name, delimiter=",")[::2, :3].copy()
        # Convert points to a NumPy array
        points_array = np.asarray(points)

        # Create a Plotly figure for the 3D scatter plot
        fig = go.Figure()

        # Add the 3D scatter plot to the figure
        fig.add_trace(
            go.Scatter3d(
                x=points_array[:, 0],
                y=points_array[:, 1],
                z=points_array[:, 2],
                mode='markers',
                marker=dict(size=1),
            )
        )

    # Convert the Plotly figure to an image
    image_bytes = fig.to_image(format="png")

    # Convert the image to Gradio-compatible format
    image = Image.open(io.BytesIO(image_bytes))

    return image