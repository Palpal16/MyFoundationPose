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
from pose_metrics import adi_est

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

def resize_mesh(mesh, new_diameter, reverse=False, diameter=None):
    '''
    It centers and resizes the centered mesh.
    If reverse, it will rotate the mesh. (Used in the first step to change the coordinate system)
    '''
    out_mesh = mesh.copy()
    if diameter is None:
      diameter, center = compute_mesh_diameter_and_center(out_mesh.vertices, 10000)
      out_mesh.vertices -= center
      logging.info(f"original diameter: {diameter}, new diameter: {new_diameter}")

    out_mesh.vertices *= new_diameter / diameter
    if reverse:
      # rot_matrix = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
      rot_matrix = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
      out_mesh.apply_transform(rot_matrix)
    return out_mesh, new_diameter

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


def estimate_and_scale_mesh(mesh, reader, max_diameter=0.3, scale_factor=1.25, additional_scale=1.2, cheating_scale=True):  ##### This values should be changed
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
    if cheating_scale:
        
        gt_diameter, _ = compute_mesh_diameter_and_center(reader.get_gt_mesh().vertices)
        logging.info(f'gt_diam is : {gt_diameter}')
        rescaled_mesh, rescaled_diameter = resize_mesh(mesh, new_diameter=gt_diameter, reverse=True)
        logging.info(f"scaled_mesh diameter: {rescaled_diameter}")
    else:
        depth = reader.get_depth(0)
        mask = reader.get_mask(0).astype(bool)
        guessed_mesh_diameter = estimate_max_length(depth, mask, reader.K)
        logging.info(f"Depth + mask diameter, without additional scalining: {guessed_mesh_diameter:.4f}m")

        # Apply constraints and scaling
        guessed_mesh_diameter = min(guessed_mesh_diameter * scale_factor, max_diameter)
        guessed_mesh_diameter = guessed_mesh_diameter * additional_scale
        
        logging.info(f"Estimated mesh diameter from depth: {guessed_mesh_diameter:.4f}m")
        
        # Scale mesh to estimated diameter
        scaled_mesh, scaled_diameter = resize_mesh(mesh, new_diameter=guessed_mesh_diameter, reverse=False)   ##### Check the reverse
        
        logging.info(f"Mesh scaled to diameter: {compute_mesh_diameter(scaled_mesh):.4f}m")   ##### Magari cambia compute_mesh con l'altra funzione
        
        true_mesh_diameter = guessed_mesh_diameter **2 / compute_mesh_diameter(scaled_mesh)
        rescaled_mesh, rescaled_diameter = resize_mesh(scaled_mesh, new_diameter=true_mesh_diameter, reverse=True)
        logging.info(f"scaled_mesh diameter: {compute_mesh_diameter(rescaled_mesh)}")

    return rescaled_mesh, rescaled_diameter

def my_depth2xyzmap(depth, K, clip_scale=1.5):
    #Creates the pointcloud from the depth
    #Removes the points that are too far (threshold = clip_scale * median_depth)
    H, W = depth.shape[:2]
    valid_mask = (depth >= 0.05)

    valid_depths = depth[valid_mask]
    median_depth = np.median(valid_depths)
    threshold = clip_scale * median_depth
    
    depth_mask = valid_mask & (np.abs(depth) <= threshold)
    
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs, ys, zs), axis=1)  # (N, 3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[~depth_mask] = 0
    
    # Optional: print statistics
    num_valid = np.sum(valid_mask)
    num_kept = np.sum(depth_mask)
    num_clipped = num_valid - num_kept
    print(f"Median depth: {median_depth:.3f}, Threshold: {threshold:.3f}")
    print(f"Clipped {num_clipped}/{num_valid} points ({100*num_clipped/num_valid:.1f}%)")
    
    return xyz_map

def rgb_depth_to_mesh_frame(pose, reader, frame_idx, boundary_distance_px=35):
    """Convert RGB-D image to point cloud in mesh frame, excluding pixels near mask boundary
    
    Args:
        pose: 4x4 object-in-camera pose matrix
        reader: Data reader object
        frame_idx: Frame index to process
        boundary_distance_px: minimum distance from boundary in pixels (default: 30)
    
    Returns:
        points_in_mesh: (N,3) 3D points in mesh frame
        colors: (N,3) corresponding RGB colors
    """
    rgb = reader.get_color(frame_idx)
    depth = reader.get_depth(frame_idx)
    mask = reader.get_mask(frame_idx).astype(bool)
    depth[~mask]=0

    # Convert depth to 3D points in camera frame
    xyz_map = my_depth2xyzmap(depth, reader.K)  # (H,W,3)

    # Compute distance transform: each pixel = distance to nearest mask boundary
    dist_transform = distance_transform_edt(mask)
    
    # Evaluate the confidence based on the distance (interior points have more confidence)
    interior_mask = dist_transform > boundary_distance_px
    mask_without_edge = dist_transform > 12
    confidence_values = np.zeros(depth.shape, dtype=np.float32)
    confidence_values[mask_without_edge]=0.1
    confidence_values[interior_mask]=0.3
    
    # Extract valid points from interior
    valid = depth > 0.001
    points_cam = xyz_map[valid]  # (N,3)
    colors = rgb[valid]  # (N,3)
    confidence_values = confidence_values[valid]  # (N)
    
    # Transform from camera frame to mesh frame
    cam_in_object = np.linalg.inv(pose)
    points_in_mesh = transform_pts(points_cam, cam_in_object)
    
    return points_in_mesh, colors, confidence_values

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


class MeshWithConfidence:
    """Mesh wrapper that tracks vertex confidence for confidence-weighted updates"""
    
    def __init__(self, mesh):
        """Initialize mesh with confidence tracking
        Args:
            mesh: trimesh object
        """
        self.mesh = mesh.copy()
        
        # Confidence tracking (0 = unobserved, 1 = frozen/certain)
        self.confidence = np.zeros(len(mesh.vertices), dtype=np.float32)
        
        # Observed data (accumulated but not yet committed to mesh)
        self.observed_positions = self.mesh.vertices.copy()
        self.observed_colors = self.mesh.visual.vertex_colors[:, :3].copy().astype(np.float32)
    
    def save(self, path):
        """Save all components separately"""
        os.makedirs(path, exist_ok=True)
        self.mesh.export(f'{path}/mesh.obj')
        np.save(f'{path}/confidence.npy', self.confidence)
        np.save(f'{path}/positions.npy', self.observed_positions)
        np.save(f'{path}/colors.npy', self.observed_colors)
    
    @classmethod
    def load(cls, path):
        """Load from saved components"""
        cmesh = cls(trimesh.load(f'{path}/mesh.obj'))
        cmesh.confidence = np.load(f'{path}/confidence.npy')
        cmesh.observed_positions = np.load(f'{path}/positions.npy')
        cmesh.observed_colors = np.load(f'{path}/colors.npy')
        return cmesh


def update_mesh_from_pointcloud(CMesh, point_cloud_xyz, point_cloud_rgb, point_cloud_conf, indices, dissimilarity_threshold=0.02):
    """Update mesh vertices with averaged positions and colors from assigned point cloud points.
    
    Args:
        CMesh: MeshWithConfidence object
        point_cloud_xyz: (N, 3) point cloud coordinates
        point_cloud_rgb: (N, 3) point cloud colors [0-255] or [0-1]
        point_cloud_conf: (N,) confidence values
        indices: (N,) indices from KDTree query
        dissimilarity_threshold: max distance to accept observation (default: 0.02m)
    """
    n_vertices = len(CMesh.mesh.vertices)
    
    # Accumulate sums and counts for each vertex
    vertex_xyz_sum = np.zeros((n_vertices, 3))
    vertex_rgb_sum = np.zeros((n_vertices, 3))
    vertex_conf_sum = np.zeros(n_vertices)
    vertex_counts = np.zeros(n_vertices)
    
    # Accumulate point cloud values to their assigned vertices
    np.add.at(vertex_xyz_sum, indices, point_cloud_xyz)
    np.add.at(vertex_rgb_sum, indices, point_cloud_rgb)
    np.add.at(vertex_conf_sum, indices, point_cloud_conf)
    np.add.at(vertex_counts, indices, 1)
    
    # Find vertices with at least one assignment
    assigned_mask = vertex_counts > 0
    
    # Take the mean
    vertex_xyz_sum[assigned_mask] = vertex_xyz_sum[assigned_mask] / vertex_counts[assigned_mask, None]
    vertex_colors = np.zeros((n_vertices, 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255  # Alpha channel
    vertex_colors[assigned_mask, :3] = (vertex_rgb_sum[assigned_mask] / vertex_counts[assigned_mask, None]).astype(np.uint8)
    vertex_conf_sum[assigned_mask] = vertex_conf_sum[assigned_mask] / vertex_counts[assigned_mask]

    # Create masks for different update strategies
    replace_mask = assigned_mask & (CMesh.confidence == 0.)
    update_mask = assigned_mask & (CMesh.confidence > 0) & (CMesh.confidence < 1)

    # REPLACE: First time observing these vertices - update observed data
    CMesh.observed_positions[replace_mask] = vertex_xyz_sum[replace_mask]
    CMesh.observed_colors[replace_mask] = vertex_colors[replace_mask, :3].astype(np.float32)
    CMesh.confidence[replace_mask] = vertex_conf_sum[replace_mask]

    # UPDATE: Previously observed vertices
    if np.any(update_mask):
        # Compute distance between current observed estimate and new observation
        distances = np.linalg.norm(
            CMesh.observed_positions[update_mask] - vertex_xyz_sum[update_mask], 
            axis=1
        )
        
        # Create full-size dissimilarity mask
        dissimilar_mask = np.zeros(n_vertices, dtype=bool)
        dissimilar_mask[update_mask] = distances > dissimilarity_threshold
        
        # Similar mask: vertices that need updating AND are similar
        similar_mask = update_mask & ~dissimilar_mask
        
        # Handle dissimilar observations
        if np.any(dissimilar_mask):
            CMesh.confidence[dissimilar_mask] -= vertex_conf_sum[dissimilar_mask]
            CMesh.confidence[dissimilar_mask] = np.maximum(0, CMesh.confidence[dissimilar_mask])
            logging.info(f"Rejected {dissimilar_mask.sum()} dissimilar observations (distance > {dissimilarity_threshold}m)")
        
        # Handle similar observations: weighted average in observed data
        if np.any(similar_mask):
            # Compute total confidence
            total_conf = CMesh.confidence[similar_mask] + vertex_conf_sum[similar_mask]
            
            # Weighted average for observed positions
            CMesh.observed_positions[similar_mask] = (
                CMesh.observed_positions[similar_mask] * CMesh.confidence[similar_mask, None] + 
                vertex_xyz_sum[similar_mask] * vertex_conf_sum[similar_mask, None]
            ) / total_conf[:, None]
            
            # Weighted average for observed colors
            CMesh.observed_colors[similar_mask] = (
                CMesh.observed_colors[similar_mask] * CMesh.confidence[similar_mask, None] + 
                vertex_colors[similar_mask, :3].astype(np.float32) * vertex_conf_sum[similar_mask, None]
            ) / total_conf[:, None]
            
            # Update confidence
            CMesh.confidence[similar_mask] = total_conf
    
    # Clamp confidence values
    CMesh.confidence[CMesh.confidence > 0.9] = 1.0
    CMesh.confidence[CMesh.confidence < 0.1] = 0.0
    
    # Commit to mesh: vertices that reached confidence = 1.0
    frozen_mask = CMesh.confidence >= 1.0
    CMesh.mesh.vertices[frozen_mask] = CMesh.observed_positions[frozen_mask]
    CMesh.mesh.visual.vertex_colors[frozen_mask, :3] = CMesh.observed_colors[frozen_mask].astype(np.uint8)
    
    n_updated = replace_mask.sum() + update_mask.sum()
    n_frozen = (CMesh.confidence >= 1.0).sum()
    logging.info(f"Updated {n_updated}/{n_vertices} vertices ({n_frozen} frozen)")

    # For the initial frames, when the mesh is not modified yet, commit the highest-confident points (interio points)
    if n_frozen == 0:
        interior_mask = (CMesh.confidence == CMesh.confidence.max())
        CMesh.mesh.vertices[interior_mask] = CMesh.observed_positions[interior_mask]
        CMesh.mesh.visual.vertex_colors[interior_mask, :3] = CMesh.observed_colors[interior_mask].astype(np.uint8)
        logging.info(f"Committed {interior_mask.sum()} interior points to mesh")
    
    return CMesh


def perform_attachment(est, CMesh, pose, reader, frame_idx):
    """Perform mesh attachment/refinement using current frame observations
    
    Args:
        est: FoundationPose estimator
        Cmesh: Current MeshWithConfidence
        pose: Current pose estimate
        reader: Data reader
        frame_idx: Current frame index
        
    Returns:
        updated_mesh: Refined mesh
    """
    logging.info(f"Performing attachment at frame {frame_idx}")

    # Convert observations to mesh frame
    points_obs, colors_obs, confidence_obs = rgb_depth_to_mesh_frame(pose, reader, frame_idx)
    
    # Find closest mesh vertices
    tree = KDTree(CMesh.mesh.vertices)
    distances, indices = tree.query(points_obs)

    # Update mesh with observations (pass frame_idx)
    CMesh = update_mesh_from_pointcloud(CMesh, points_obs, colors_obs, confidence_obs, indices)

    # Smooth the attached mesh
    CMesh.mesh = smooth_mesh_taubin(CMesh.mesh)
    
    # Reset estimator with new mesh
    est.reset_object(
        model_pts=CMesh.mesh.vertices,
        model_normals=CMesh.mesh.vertex_normals,
        mesh=CMesh.mesh
    )
    
    return CMesh

def pose_to_Rt(pose):
    R = pose[:3, :3]
    t = pose[:3, 3:4]
    return R, t

def evaluate_frame(gt_mesh, gt_pose, est_mesh, est_pose):
    pts_gt_orig = np.array(gt_mesh.vertices, dtype=np.float32)
    pts_est_orig = np.array(est_mesh.vertices, dtype=np.float32)
    R_est, t_est = pose_to_Rt(est_pose)
    R_gt, t_gt = pose_to_Rt(gt_pose)
    frame_metrics = {}
    frame_metrics['3D_IOU'], frame_metrics['ADI'] = adi_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig)
    return frame_metrics

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--mesh_file', type=str, default='/Experiments/simonep01/ho3d/first_frame_instantmeshes/AP14/mesh.obj')
    #parser.add_argument('--mesh_file', type=str, default='/home/simonep01/sam-3d-objects/meshes/MPM10/transformed_mesh.obj')
    parser.add_argument('--mesh_file', type=str, default=None)
    parser.add_argument('--video_id', type=str, default='SM1')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=3)
    parser.add_argument('--debug_dir', type=str, default='debug/attach')
    parser.add_argument('--n_frames', type=int, default=None)
    parser.add_argument('--attach_every_n_frames', type=int, default=8, help='Perform mesh attachment every N frames (0 = disabled, 1 = every frame, 2 = every other frame, etc.)')
    parser.add_argument('--evaluation', action='store_false')
    args = parser.parse_args()

    test_scene_dir= f'/Experiments/simonep01/ho3d/evaluation/{args.video_id}'
    reader = Ho3dReader(video_dir=test_scene_dir)

    if args.n_frames==None:
        n_frames = len(reader.color_files)
    else:
        n_frames = max(args.n_frames, len(reader.color_files))

    if args.mesh_file==None:
        mesh_file = f'/home/simonep01/sam-3d-objects/meshes/{args.video_id}/reduced_mesh.obj'
    else:
        mesh_file = args.mesh_file


    debug_dir = f'{args.debug_dir}/{args.video_id}'
    os.makedirs(debug_dir, exist_ok=True)

    set_logging_format()
    set_seed(0)

    debug = args.debug
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    log_path = os.path.join(debug_dir, 'log.txt')
    f = open(log_path, "w")

    mesh = trimesh.load(mesh_file)
    
    mesh, _ = estimate_and_scale_mesh(mesh,reader)

    CMesh = MeshWithConfidence(mesh)
    
    to_origin, extents = trimesh.bounds.oriented_bounds(CMesh.mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=CMesh.mesh.vertices, model_normals=CMesh.mesh.vertex_normals, mesh=CMesh.mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    if args.evaluation:
        eval_dir = f"{debug_dir}/evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        gt_mesh = reader.get_gt_mesh()
        metrics_keys = ['ADI', '3D_IOU']
        per_frame_metrics = {key: [] for key in metrics_keys}

    for i in range(n_frames):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i==0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
        
        if args.evaluation:
            frame_metrics = evaluate_frame(gt_mesh, reader.get_gt_pose(i), CMesh.mesh, pose)
            for key in metrics_keys:
                per_frame_metrics[key].append(frame_metrics[key])
        
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
            
        if debug>=1 and i%20==0:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            
        if debug>=2 and i%20==0:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        if debug >=3:
            if i in [0,5,20,50,100,200,400,800, 1000, 1300, 1600]:
                CMesh.save(f'{debug_dir}/debug/mesh_{i}')
                if args.evaluation and i>50:
                    summary = {}
                    for key in metrics_keys:
                        summary[key] = {
                            'mean': float(np.mean(per_frame_metrics[key])),
                            'min': float(np.min(per_frame_metrics[key])),
                            'max': float(np.max(per_frame_metrics[key]))
                        }
                        output_file = os.path.join(eval_dir, f'{key}_per_frame.json')
                        with open(output_file, 'w') as f:
                            json.dump([float(x) for x in per_frame_metrics[key]], f, indent=2)

                    summary_file = os.path.join(eval_dir, 'summary.json')
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)

        if args.attach_every_n_frames > 0 and i % args.attach_every_n_frames == 0:
            CMesh = perform_attachment(est, CMesh, pose, reader, i)
                

    
    CMesh.mesh.export(f'{debug_dir}/final_mesh.obj')
    
    if args.evaluation:        
        summary = {}
        for key in metrics_keys:
            summary[key] = {
                'mean': float(np.mean(per_frame_metrics[key])),
                'min': float(np.min(per_frame_metrics[key])),
                'max': float(np.max(per_frame_metrics[key]))
            }

            output_file = os.path.join(eval_dir, f'{key}_per_frame.json')
            with open(output_file, 'w') as f:
                json.dump([float(x) for x in per_frame_metrics[key]], f, indent=2)

        summary_file = os.path.join(eval_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Evaluation Results ({n_frames} frames)")
        print(f"{'='*60}")
        print(f"ADI (Average Distance):        {summary['ADI']['mean']:.4f} mm")
        print(f"3D IOU:                        {summary['3D_IOU']['mean']:.3f} %")

    logging.info("Processing complete")
    f.close()