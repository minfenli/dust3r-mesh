import torch
import numpy as np
import open3d as o3d
import trimesh

def features_to_world_space_mesh(world_space_points, colors, mask=None, edge_threshold=-1, min_triangles_connected=-1):
    """
    project features to mesh in world space and return (vertices, faces, colors) result by applying simple triangulation from image-structure.

    :param colors: (C, H, W)
    :param depth: (H, W)
    :param fov_in_degrees: fov_in_degrees
    :param world_to_cam: (4, 4)
    :param mask: (H, W)
    :param edge_threshold: only save faces that no any edge are larger than others. Default: -1 (:= do not apply threshold)
    """

    # get point cloud from depth map
    C, H, W = colors.shape
    colors = colors.reshape(C, -1)
    world_space_points = world_space_points.reshape(3, -1)

    # define vertex_ids for triangulation
    '''
    00---01
    |    |
    10---11
    '''
    vertex_ids = torch.arange(H*W).reshape(H, W).to(colors.device)
    vertex_00 = remapped_vertex_00 = vertex_ids[:H-1, :W-1]
    vertex_01 = remapped_vertex_01 = (remapped_vertex_00 + 1)
    vertex_10 = remapped_vertex_10 = (remapped_vertex_00 + W)
    vertex_11 = remapped_vertex_11 = (remapped_vertex_00 + W + 1)

    if mask is not None:
        def dilate(x, k=3):
            x = torch.nn.functional.conv2d(
                x.float()[None, None, ...],
                torch.ones(1, 1, k, k).to(mask.device),
                padding="same"
            )
            return x.squeeze() > 0

        # need dilated mask for "connecting vertices", e.g. face at the mask-edge connected to next masked-out vertex
        '''
        x---x---o
        | / | / |  
        x---o---o
        '''
        mask_dilated = dilate(mask, k=5)

        # only keep vertices/features for faces that need to be added (i.e. are masked) -- rest of the faces are already present in 3D
        colors = colors[:, mask_dilated.flatten()]
        world_space_points = world_space_points[:, mask_dilated.flatten()]

        # remap vertex id's to shortened list of vertices
        remap = torch.bucketize(vertex_ids, vertex_ids[mask_dilated])
        remap[~mask_dilated] = -1  # mark invalid vertex_ids with -1 --> due to dilation + triangulation, a few faces will contain -1 values --> need to filter them
        remap = remap.flatten()
        mask_dilated = mask_dilated[:H-1, :W-1]
        vertex_00 = vertex_00[mask_dilated]
        vertex_01 = vertex_01[mask_dilated]
        vertex_10 = vertex_10[mask_dilated]
        vertex_11 = vertex_11[mask_dilated]
        remapped_vertex_00 = remap[vertex_00]
        remapped_vertex_01 = remap[vertex_01]
        remapped_vertex_10 = remap[vertex_10]
        remapped_vertex_11 = remap[vertex_11]

    # triangulation: upper-left and lower-right triangles from image structure
    faces_upper_left_triangle = torch.stack(
        [remapped_vertex_00.flatten(), remapped_vertex_10.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )
    faces_lower_right_triangle = torch.stack(
        [remapped_vertex_10.flatten(), remapped_vertex_11.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )

    # filter faces with -1 vertices and combine
    mask_upper_left = torch.all(faces_upper_left_triangle >= 0, dim=0)
    faces_upper_left_triangle = faces_upper_left_triangle[:, mask_upper_left]
    mask_lower_right = torch.all(faces_lower_right_triangle >= 0, dim=0)
    faces_lower_right_triangle = faces_lower_right_triangle[:, mask_lower_right]
    faces = torch.cat([faces_upper_left_triangle, faces_lower_right_triangle], dim=1)

    # clean mesh
    world_space_points, faces, colors = clean_mesh(
        world_space_points,
        faces,
        colors,
        edge_threshold=edge_threshold,
        min_triangles_connected=min_triangles_connected,
        fill_holes=True
    )

    return world_space_points, faces, colors

def edge_threshold_filter(vertices, faces, edge_threshold=0.5):
    """
    Only keep faces where all edges are smaller than edge length multiplied by edge_threshold.
    Will remove stretch artifacts that are caused by inconsistent depth at object borders

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param edge_threshold: maximum length per edge (otherwise removes that face).

    :return: filtered faces
    """

    p0, p1, p2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
    d01 = torch.linalg.vector_norm(p0 - p1, dim=0)
    d02 = torch.linalg.vector_norm(p0 - p2, dim=0)
    d12 = torch.linalg.vector_norm(p1 - p2, dim=0)

    d_mean = (d01 + d02 + d12) * edge_threshold

    mask_small_edge = (d01 < d_mean) * (d02 < d_mean) * (d12 < d_mean)
    faces = faces[:, mask_small_edge]

    return faces

def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, edge_threshold: float = 0.1, min_triangles_connected: int = -1, fill_holes: bool = True):
    """
    Performs the following steps to clean the mesh:

    1. edge_threshold_filter
    2. remove_duplicated_vertices, remove_duplicated_triangles, remove_degenerate_triangles
    3. remove small connected components
    4. remove_unreferenced_vertices
    5. fill_holes

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param fill_holes: If true, will perform trimesh fill_holes step, otherwise not.

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    if edge_threshold > 0:
        # remove long edges
        faces = edge_threshold_filter(vertices, faces, edge_threshold)

    # cleanup via open3d
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    if min_triangles_connected > 0:
        # remove small components via open3d
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles_connected
        mesh.remove_triangles_by_mask(triangles_to_remove)

    # cleanup via open3d
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        # misc cleanups via trimesh
        mesh = o3d_to_trimesh(mesh)
        mesh.process()
        mesh.fill_holes()

        return trimesh_to_torch(mesh, v=vertices, f=faces, c=colors)
    else:
        return o3d_mesh_to_torch(mesh, v=vertices, f=faces, c=colors)


def torch_to_o3d_mesh(vertices, faces, colors):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.T.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return mesh


def o3d_mesh_to_torch(mesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.triangles)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).T
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def torch_to_o3d_pcd(vertices, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return pcd


def o3d_pcd_to_torch(pcd, p=None, c=None):
    points = torch.from_numpy(np.asarray(pcd.points)).T
    if p is not None:
        points = points.to(p)
    colors = torch.from_numpy(np.asarray(pcd.colors)).T
    if c is not None:
        colors = colors.to(c)
    return points, colors


def torch_to_trimesh(vertices, faces, colors):
    mesh = trimesh.base.Trimesh(
        vertices=vertices.T.cpu().numpy(),
        faces=faces.T.cpu().numpy(),
        vertex_colors=(colors.T.cpu().numpy() * 255).astype(np.uint8),
        process=False)

    return mesh


def trimesh_to_torch(mesh: trimesh.base.Trimesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.faces)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors, dtype=float) / 255).T[:3]
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.base.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors).clip(0, 1) * 255).astype(np.uint8),
        process=False)

def save_mesh(vertices, faces, colors, target_path):
    colors = colors[:3, ...]
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(target_path, mesh, compressed=True, write_vertex_colors=True, print_progress=True)