# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from estimater import *
from datareader import *
import argparse
from scipy.spatial import KDTree
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist

def estimate_max_length(depth, mask, K):
    """
    Estimate the maximum length of the projected point cloud from depth, mask, and intrinsic matrix K.
    
    Parameters:
    - depth: (H, W) array representing the depth image
    - mask: (H, W) array representing the binary mask (1 for valid points, 0 for invalid)
    - K: (3, 3) array representing the intrinsic camera matrix
    
    Returns:
    - max_length: Maximum length (distance) between two points in the projected point cloud
    """
    
    # Get the height and width of the image
    H, W = depth.shape

    # Generate pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten arrays for easier manipulation
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth.flatten()
    mask_flat = mask.flatten()

    # Filter out points outside the mask (invalid points)
    valid_indices = (mask_flat > 0) & (depth_flat >= 0.1)
    u_flat = u_flat[valid_indices]
    v_flat = v_flat[valid_indices]
    depth_flat = depth_flat[valid_indices]

    # Apply depth threshold (10cm or large depth)
    median_depth = np.median(depth_flat)
    # depth_valid_indices = (depth_flat < median_depth+0.5) & (depth_flat >= 0.1)
    # u_flat = u_flat[depth_valid_indices]
    # v_flat = v_flat[depth_valid_indices]
    # depth_flat = depth_flat[depth_valid_indices]

    # Get intrinsic camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Reproject pixel coordinates into 3D space
    X = (u_flat - cx) * median_depth / fx
    Y = (v_flat - cy) * median_depth / fy
    Z = depth_flat

    # Compute pairwise distances between points in the X-Y plane (real-world length along the screen)
    # We only consider X and Y here, ignoring Z for "along the screen" distances
    points_2d_real_world = np.vstack((X, Y)).T

    # Compute pairwise distances between 2D real-world points
    distances_2d_real_world = pdist(points_2d_real_world)

    # Find the maximum real-world distance along the screen
    max_length_real_world = np.max(distances_2d_real_world)

    return max_length_real_world

def compute_mesh_diameter_and_center(model_pts, n_sample=10000):
  if n_sample is None:
    pts = model_pts
  else:
    ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
    pts = model_pts[ids]
  dists = np.linalg.norm(pts[None]-pts[:,None], axis=-1)
  diameter = dists.max()
  return diameter, model_pts.mean(0)

def resize_mesh(mesh, new_diameter, reverse=False, diameter=None, mesh_unseen=None):
    '''
    It centers and resizes the centered mesh.
    If reverse, it will rotate the mesh. (Used in the first step to change the coordinate system)
    '''
    if diameter is None:
      diameter, center = compute_mesh_diameter_and_center(mesh.vertices, 10000)
      mesh.vertices -= center
      if mesh_unseen is not None:
        mesh_unseen.vertices -= center
      print(f"original diameter: {diameter}, new diameter: {new_diameter}")

    mesh.vertices *= new_diameter / diameter
    if mesh_unseen is not None:
      mesh_unseen.vertices *= new_diameter / diameter
    if reverse:
      # rot_matrix = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
      rot_matrix = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
      mesh.apply_transform(rot_matrix)
      if mesh_unseen is not None:
        mesh_unseen.apply_transform(rot_matrix)
    if mesh_unseen is not None:
      return mesh, mesh_unseen, new_diameter
    else:
      return mesh, None, new_diameter

def compute_mesh_diameter(mesh):
    """Compute mesh diameter using SVD
    
    Args:
        mesh: trimesh object or path to mesh file
        
    Returns:
        diameter: float
    """
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)
    
    import scipy.linalg
    u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
    pts = u @ np.diag(s)
    diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    return float(diameter)


def estimate_and_scale_mesh(mesh, reader, max_diameter=0.3, scale_factor=1.25, additional_scale=1.2):
    """Estimate mesh scale from depth observation and resize mesh accordingly
    
    Args:
        mesh: trimesh object
        reader: Data reader object
        max_diameter: maximum expected diameter in meters (default: 0.3)
        scale_factor: initial scaling factor (default: 1.25)
        additional_scale: additional scaling factor (default: 1.2)
        
    Returns:
        scaled_mesh: trimesh object scaled to estimated size
        estimated_diameter: estimated diameter
    """
    # Estimate diameter from depth observation
    depth = reader.get_depth(0)
    mask = reader.get_mask(0).astype(bool)
    guessed_mesh_diameter = estimate_max_length(depth, mask, reader.K)
    
    # Apply constraints and scaling
    guessed_mesh_diameter = min(guessed_mesh_diameter * scale_factor, max_diameter)
    guessed_mesh_diameter = guessed_mesh_diameter * additional_scale
    
    logging.info(f"Estimated mesh diameter from depth: {guessed_mesh_diameter:.4f}m")
    
    # Scale mesh to estimated diameter
    scaled_mesh, _, actual_diameter = resize_mesh(mesh, new_diameter=guessed_mesh_diameter, reverse=True)
    
    logging.info(f"Mesh scaled to diameter: {compute_mesh_diameter(scaled_mesh):.4f}m")
    
    return scaled_mesh, guessed_mesh_diameter


def rgb_depth_to_mesh_frame(pose, reader, frame_idx, boundary_distance_px=10):
    """Convert RGB-D image to point cloud in mesh frame, excluding pixels near mask boundary
    
    Args:
        pose: 4x4 object-in-camera pose matrix
        reader: Data reader object
        frame_idx: Frame index to process
        boundary_distance_px: minimum distance from boundary in pixels (default: 10)
    
    Returns:
        points_in_mesh: (N,3) 3D points in mesh frame
        colors: (N,3) corresponding RGB colors
    """
    rgb = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    mask = reader.get_mask(frame_idx).astype(bool)
    
    # Compute distance transform: each pixel = distance to nearest boundary
    dist_transform = distance_transform_edt(mask)
    
    # Keep only pixels sufficiently far from boundary
    interior_mask = dist_transform > boundary_distance_px
    
    # Zero out depth outside interior region
    depth[~interior_mask] = 0
    
    # 1. Convert depth to 3D points in camera frame
    xyz_map = depth2xyzmap(depth, reader.K)  # (H,W,3)
    
    # 2. Extract valid points from interior
    valid = depth > 0.1
    points_cam = xyz_map[valid]  # (N,3)
    colors = rgb[valid]  # (N,3)
    
    # 3. Transform from camera frame to mesh frame
    cam_in_object = np.linalg.inv(pose)
    points_in_mesh = transform_pts(points_cam, cam_in_object)
    
    return points_in_mesh, colors


def update_mesh_from_pointcloud(mesh_in, point_cloud_xyz, point_cloud_rgb, indices):
    """Update mesh vertices with averaged positions and colors from assigned point cloud points.
    
    Args:
        mesh_in: trimesh object
        point_cloud_xyz: (N, 3) point cloud coordinates
        point_cloud_rgb: (N, 3) point cloud colors [0-255] or [0-1]
        indices: (N,) indices from KDTree query
    """
    mesh = mesh_in.copy()
    n_vertices = len(mesh.vertices)
    
    # Accumulate sums and counts for each vertex
    vertex_xyz_sum = np.zeros((n_vertices, 3))
    vertex_rgb_sum = np.zeros((n_vertices, 3))
    vertex_counts = np.zeros(n_vertices)
    
    # Accumulate point cloud values to their assigned vertices
    np.add.at(vertex_xyz_sum, indices, point_cloud_xyz)
    np.add.at(vertex_rgb_sum, indices, point_cloud_rgb)
    np.add.at(vertex_counts, indices, 1)
    
    # Find vertices with at least one assignment
    assigned_mask = vertex_counts > 0
    
    # Update positions
    mesh.vertices[assigned_mask] = vertex_xyz_sum[assigned_mask] / vertex_counts[assigned_mask, None]
    
    # Update colors
    vertex_colors = np.zeros((n_vertices, 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255  # Alpha channel
    vertex_colors[assigned_mask, :3] = (vertex_rgb_sum[assigned_mask] / vertex_counts[assigned_mask, None]).astype(np.uint8)
    
    mesh.visual.vertex_colors[assigned_mask] = vertex_colors[assigned_mask]
    
    logging.info(f"Updated {assigned_mask.sum()}/{len(mesh.vertices)} vertices")
    
    return mesh


def smooth_mesh_taubin(mesh, iterations=10, lambda_factor=0.5, mu_factor=-0.53):
    """Apply Taubin smoothing to mesh vertices
    
    Taubin smoothing prevents volume shrinkage by alternating positive and 
    negative smoothing steps. Better than pure Laplacian for noisy data.
    
    Args:
        mesh: trimesh object
        iterations: number of smoothing iterations (default: 10)
        lambda_factor: positive smoothing factor, typically 0.3-0.6 (default: 0.5)
        mu_factor: negative smoothing factor, typically -0.5 to -0.55 (default: -0.53)
    
    Returns:
        smoothed mesh (trimesh object)
    """
    mesh = mesh.copy()
    vertices = mesh.vertices.copy()
    n_vertices = len(vertices)
    
    # Build adjacency matrix from edges
    edges = mesh.edges_unique
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row))
    adjacency = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    
    # Compute degree (number of neighbors) per vertex
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # avoid division by zero for isolated vertices
    
    # Normalize adjacency to get averaging weights
    degree_inv = 1.0 / degree
    weight_matrix = adjacency.multiply(degree_inv[:, None])
    
    for _ in range(iterations):
        # Step 1: Positive smoothing (lambda)
        laplacian = weight_matrix @ vertices - vertices
        vertices += lambda_factor * laplacian
        
        # Step 2: Negative smoothing (mu) - prevents shrinkage
        laplacian = weight_matrix @ vertices - vertices
        vertices += mu_factor * laplacian
    
    mesh.vertices = vertices
    return mesh


def perform_attachment(est, mesh, pose, reader, frame_idx):
    """Perform mesh attachment/refinement using current frame observations
    
    Args:
        est: FoundationPose estimator
        mesh: Current mesh
        pose: Current pose estimate
        reader: Data reader
        frame_idx: Current frame index
        
    Returns:
        updated_mesh: Refined mesh
    """
    logging.info(f"Performing attachment at frame {frame_idx}")
    
    # Convert observations to mesh frame
    points_obs, colors_obs = rgb_depth_to_mesh_frame(pose, reader, frame_idx)
    
    # Find closest mesh vertices
    tree = KDTree(mesh.vertices)
    distances, indices = tree.query(points_obs)
    
    # Update mesh with observations
    attached_mesh = update_mesh_from_pointcloud(mesh, points_obs, colors_obs, indices)
    
    # Smooth the attached mesh
    smooth_attached_mesh = smooth_mesh_taubin(attached_mesh)
    
    # Reset estimator with new mesh
    est.reset_object(
        model_pts=smooth_attached_mesh.vertices,
        model_normals=smooth_attached_mesh.vertex_normals,
        mesh=smooth_attached_mesh
    )
    
    return smooth_attached_mesh


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mesh_file', type=str, default='/Experiments/simonep01/demo_data/light_ho3d/evaluation/AP14/instantmesh/mesh.obj')
    parser.add_argument('--mesh_file', type=str, default='/Experiments/simonep01/demo_data/ho3d_first_frame_instantmeshes/AP14/mesh.obj')
    parser.add_argument('--test_scene_dir', type=str, default='/Experiments/simonep01/demo_data/light_ho3d/evaluation/AP14')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default='debug')
    parser.add_argument('--attach_every_n_frames', type=int, default=2, help='Perform mesh attachment every N frames (0 = disabled, 1 = every frame, 2 = every other frame, etc.)')
    parser.add_argument('--boundary_distance_px', type=int, default=20, help='Minimum distance from mask boundary in pixels for attachment (default: 10)')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)

    reader = Ho3dReader(video_dir=args.test_scene_dir)
    
    mesh, _ = estimate_and_scale_mesh(mesh,reader)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i==0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            
            if debug>=3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth>=0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            
            if args.attach_every_n_frames > 0:
                mesh = perform_attachment(est, mesh, pose, reader, i)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
            
            if args.attach_every_n_frames > 0 and i % args.attach_every_n_frames == 0:
                mesh = perform_attachment(est, mesh, pose, reader, i)
        
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
            
        if debug>=1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            
        if debug>=2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    if args.attach_every_n_frames > 0:
            mesh.export(f'{debug_dir}/attached_mesh.obj')