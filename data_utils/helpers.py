import random
import torch
import numpy as np
from stl import mesh
import open3d as o3

def seed_everything_deterministic(seed):
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If using GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as err:
        print(err)

def pc_normalize(pc, return_scale=False):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_scale:
        return pc, m
    return pc


def get_points_from_triangles(triangles, flip_axis=-1):
    points_A = triangles[:, 0:3]
    points_B = triangles[:, 3:6]
    points_C = triangles[:, 6:9]

    if flip_axis > -1:
        points_A[:, flip_axis] = -points_A[:, flip_axis]
        points_B[:, flip_axis] = -points_B[:, flip_axis]
        points_C[:, flip_axis] = -points_C[:, flip_axis]

    return points_A, points_B, points_C


def stl_to_xyz_with_normals_vectorized(
    input_stl_file, output_xyz_file=None, sep=",", stride=1, flip_axis=-1, permutate=False,
    with_normals=True
):
    stl_mesh = mesh.Mesh.from_file(input_stl_file)
    triangles = stl_mesh.vectors.reshape(-1, 9)  # Reshape to (N, 9) array
    triangles = triangles[::stride, :]

    # Calculate normals for all triangles
    points_A, points_B, points_C = get_points_from_triangles(triangles, flip_axis)
    edges1 = points_B - points_A
    edges2 = points_C - points_A
    if with_normals:
        normals = np.cross(edges1, edges2)
        if flip_axis > -1:
            normals = -normals
        norm_mags = np.linalg.norm(normals, axis=1)
        eps = 1e-8
        normals /= norm_mags[:, np.newaxis] + eps

        # Combine vertices and normals
        vertices = np.hstack((points_A, normals))
    else:
        vertices = points_A

    # permutate point cloud
    if permutate:
        np.random.shuffle(np.arange(vertices.shape[0]))

    # Save to XYZ file
    if output_xyz_file is not None:
        np.savetxt(output_xyz_file, vertices, delimiter=sep, fmt="%.8f")
    return vertices


def read_point_cloud_text(points_file, stride=1, flip_axis=-1, use_normals=True):
    cutoff = 6 if use_normals else 3
    points = np.loadtxt(points_file, delimiter=",")[::stride, :cutoff].copy()
    # transformation
    if flip_axis > -1:
        points[:, flip_axis] = -points[:, flip_axis]
        points[:, flip_axis + 3] = -points[:, flip_axis + 3]
    return points


def visualize_pointcloud(points_file, stride=1, flip_axis=-1, postprocess_fn=None):
    points = np.loadtxt(points_file, delimiter=",")[::stride, :3].copy()
    visualize_pointcloud_np(points, flip_axis, postprocess_fn)


def visualize_pointcloud_np(points, flip_axis=-1, postprocess_fn=None):
    pcd = o3.geometry.PointCloud()
    # transformation
    if flip_axis > -1:
        points[:, flip_axis] = -points[:, flip_axis]
    if postprocess_fn:
        points = postprocess_fn(points)
    # visualize
    pcd.points = o3.utility.Vector3dVector(points)
    o3.visualization.draw_plotly([pcd])

def random_point_dropout(points, max_dropout_ratio=0.875):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((points.shape[1]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        points[drop_idx,:] = points[0,:] # set to the first point
    return points

def shift_point_cloud(points, shift_range=0.1):
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    points[:,:3] = points[:,:3] + shifts
    return points


def obj_to_xyz_normals(input_path, output_path, sep=','):
    mesh = o3.io.read_triangle_mesh(input_path)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    # Combine vertices and normals into a single array
    xyz_normals = np.hstack((vertices, normals))

    # Save to XYZ file
    np.savetxt(output_path, xyz_normals, fmt='%.6f', delimiter=sep)