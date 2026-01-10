"""Pose error metrics for 6D object pose estimation.

This module contains metrics that work with:
1. Two different point clouds (estimated and ground truth can have different sampling)
   - Functions ending with '_est' suffix
2. Single point cloud (same mesh for both poses, more efficient)
   - Functions without '_est' suffix
"""

import math
import numpy as np
from scipy import spatial


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.
    
    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def project_pts(pts, K, R, t):
    """Projects 3D points.
    
    :param pts: nx3 ndarray with the 3D points.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """
    assert pts.shape[1] == 3
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T


def decompose_pose_matrix(pose):
    """Decompose 4x4 pose matrix into rotation and translation.
    
    :param pose: 4x4 ndarray with the pose matrix
    :return: tuple (R, t) where R is 3x3 rotation matrix and t is 3x1 translation vector
    """
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3:4]  # Keep as 2D array with shape (3,1)
        return R, t
    else:
        raise ValueError("Pose matrix must be 4x4")


def calculate_3d_iou(pts_est, pts_gt):
    """Calculate 3D IOU between two point clouds using their bounding boxes.
    
    Args:
        pts_est: nx3 ndarray of estimated 3D points
        pts_gt: mx3 ndarray of ground truth 3D points
    
    Returns:
        float: IOU score between 0 and 100
    """
    # Calculate min/max bounds for each point cloud
    min_est = np.min(pts_est, axis=0)
    max_est = np.max(pts_est, axis=0)
    min_gt = np.min(pts_gt, axis=0)
    max_gt = np.max(pts_gt, axis=0)
    
    # Calculate volumes
    vol_est = np.prod(max_est - min_est)
    vol_gt = np.prod(max_gt - min_gt)
    
    # Calculate intersection bounds
    min_intersection = np.maximum(min_est, min_gt)
    max_intersection = np.minimum(max_est, max_gt)
    
    # Check if boxes overlap
    if np.any(max_intersection < min_intersection):
        return 0.0
    
    # Calculate intersection volume
    vol_intersection = np.prod(np.maximum(0, max_intersection - min_intersection))
    
    # Calculate union volume
    vol_union = vol_est + vol_gt - vol_intersection
    
    # Calculate IOU
    iou = vol_intersection / vol_union
    return (iou * 100.0)


# ============================================================================
# SINGLE MESH VERSIONS (More efficient when using same mesh for both poses)
# ============================================================================

def add(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None):
    """Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).
    
    Single mesh version: Uses same point cloud for both poses (more efficient).
    Accepts either 4x4 pose matrices or separate R, t inputs.
    
    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)
    
    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")
    
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).
    
    Single mesh version: Uses same point cloud for both poses (more efficient).
    Accepts either 4x4 pose matrices or separate R, t inputs.
    
    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)
    
    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")
    
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    
    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    
    e = nn_dists.mean()
    return e


def mssd(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, pts=None, syms=None):
    """Maximum Symmetry-Aware Surface Distance (MSSD).
    
    Single mesh version: Uses same point cloud for both poses (more efficient).
    Accepts either 4x4 pose matrices or separate R, t inputs.
    
    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param pts: nx3 ndarray with 3D model points
    :param syms: Set of symmetry transformations, each given by a dictionary with:
                 - 'R': 3x3 ndarray with the rotation matrix
                 - 't': 3x1 ndarray with the translation vector
                 or
                 - 'pose': 4x4 ndarray with the pose matrix
    :return: The calculated error
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)
    
    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")
    
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    
    es = []
    for sym in syms:
        if "pose" in sym:
            R_sym, t_sym = decompose_pose_matrix(sym["pose"])
        else:
            R_sym, t_sym = sym["R"], sym["t"]
        
        R_gt_sym = R_gt.dot(R_sym)
        t_gt_sym = R_gt.dot(t_sym) + t_gt
        pts_gt_sym = transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        
        es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())
    
    return min(es)


def mspd(pose_est=None, pose_gt=None, R_est=None, t_est=None, R_gt=None, t_gt=None, K=None, pts=None, syms=None):
    """Maximum Symmetry-Aware Projection Distance (MSPD).
    
    Single mesh version: Uses same point cloud for both poses (more efficient).
    Accepts either 4x4 pose matrices or separate R, t inputs.
    
    :param pose_est: 4x4 ndarray with the estimated pose matrix (optional)
    :param pose_gt: 4x4 ndarray with the ground-truth pose matrix (optional)
    :param R_est: 3x3 ndarray with the estimated rotation matrix (optional)
    :param t_est: 3x1 ndarray with the estimated translation vector (optional)
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix (optional)
    :param t_gt: 3x1 ndarray with the ground-truth translation vector (optional)
    :param K: 3x3 ndarray with the intrinsic camera matrix
    :param pts: nx3 ndarray with 3D model points
    :param syms: Set of symmetry transformations, each given by a dictionary with:
                 - 'R': 3x3 ndarray with the rotation matrix
                 - 't': 3x1 ndarray with the translation vector
                 or
                 - 'pose': 4x4 ndarray with the pose matrix
    :return: The calculated error
    """
    # Handle pose matrix inputs
    if pose_est is not None:
        R_est, t_est = decompose_pose_matrix(pose_est)
    if pose_gt is not None:
        R_gt, t_gt = decompose_pose_matrix(pose_gt)
    
    # Verify we have all needed transforms
    if R_est is None or t_est is None or R_gt is None or t_gt is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")
    
    proj_est = project_pts(pts, K, R_est, t_est)
    
    es = []
    for sym in syms:
        if "pose" in sym:
            R_sym, t_sym = decompose_pose_matrix(sym["pose"])
        else:
            R_sym, t_sym = sym["R"], sym["t"]
        
        R_gt_sym = R_gt.dot(R_sym)
        t_gt_sym = R_gt.dot(t_sym) + t_gt
        proj_gt_sym = project_pts(pts, K, R_gt_sym, t_gt_sym)
        
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
    
    return min(es)


def calculate_3d_iou_with_pose(pose1=None, pose2=None, R1=None, t1=None, R2=None, t2=None, pts=None):
    """Calculate 3D IoU between two transformed point clouds.
    
    Single mesh version: Uses same point cloud for both poses (more efficient).
    Accepts either 4x4 pose matrices or separate R, t inputs.
    
    :param pose1: 4x4 ndarray with the first pose matrix (optional)
    :param pose2: 4x4 ndarray with the second pose matrix (optional)
    :param R1: 3x3 ndarray with the first rotation matrix (optional)
    :param t1: 3x1 ndarray with the first translation vector (optional)
    :param R2: 3x3 ndarray with the second rotation matrix (optional)
    :param t2: 3x1 ndarray with the second translation vector (optional)
    :param pts: nx3 ndarray with original 3D points
    :return: IoU score between 0 and 100
    """
    # Handle pose matrix inputs
    if pose1 is not None:
        R1, t1 = decompose_pose_matrix(pose1)
    if pose2 is not None:
        R2, t2 = decompose_pose_matrix(pose2)
    
    # Verify we have all needed transforms
    if R1 is None or t1 is None or R2 is None or t2 is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")
    
    # Transform points using respective poses
    transformed_pts1 = transform_pts_Rt(pts, R1, t1)
    transformed_pts2 = transform_pts_Rt(pts, R2, t2)
    
    return calculate_3d_iou(transformed_pts1, transformed_pts2)


# ============================================================================
# TWO MESH VERSIONS (For different point clouds in estimated vs ground truth)
# ============================================================================

def mssd_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).
    
    Two mesh version: Supports different point clouds for estimated and ground truth.
    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/
    
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
                 - 'R': 3x3 ndarray with the rotation matrix.
                 - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts_est_orig, R_est, t_est)
    
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = transform_pts_Rt(pts_gt_orig, R_gt_sym, t_gt_sym)
        
        # Build KD-tree for fast nearest neighbor search
        tree = spatial.cKDTree(pts_gt_sym)
        
        # Find distances to nearest neighbors
        distances, _ = tree.query(pts_est)
        
        # Maximum distance for this symmetry
        es.append(np.max(distances))
    
    return min(es)


def mspd_est(K, R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).
    
    Two mesh version: Supports different point clouds for estimated and ground truth.
    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/
    
    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
                 - 'R': 3x3 ndarray with the rotation matrix.
                 - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = project_pts(pts_est_orig, K, R_est, t_est)
    
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = project_pts(pts_gt_orig, K, R_gt_sym, t_gt_sym)
        
        # Build KD-tree for fast nearest neighbor search in 2D
        tree = spatial.cKDTree(proj_gt_sym)
        
        # Find distances to nearest neighbors
        distances, _ = tree.query(proj_est)
        
        # Maximum distance for this symmetry
        es.append(np.max(distances))
    
    return min(es)


def adi_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).
    
    Two mesh version: Supports different point clouds for estimated and ground truth.
    
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :return: Tuple (iou, error) - 3D IoU and average distance error.
    """
    # Transform points using respective poses
    pts_est = transform_pts_Rt(pts_est_orig, R_est, t_est)
    pts_gt = transform_pts_Rt(pts_gt_orig, R_gt, t_gt)
    
    iou = calculate_3d_iou(pts_est, pts_gt)
    
    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    
    e = nn_dists.mean()
    return iou, e


def chamfer_distance(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig):
    """Calculate the Chamfer distance between two point clouds.
    
    Two mesh version: Supports different point clouds for estimated and ground truth.
    
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param pts_est_orig: nx3 ndarray with 3D model points for estimated pose.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts_gt_orig: mx3 ndarray with 3D model points for ground truth pose.
    :return: The Chamfer distance (mean of both directions).
    """
    # Transform points using respective poses
    pts_est = transform_pts_Rt(pts_est_orig, R_est, t_est)
    pts_gt = transform_pts_Rt(pts_gt_orig, R_gt, t_gt)
    
    # Create KD-trees for efficient nearest neighbor search
    tree_est = spatial.cKDTree(pts_est)
    tree_gt = spatial.cKDTree(pts_gt)
    
    # Calculate distances from estimated points to nearest ground truth points
    est_to_gt_dists, _ = tree_gt.query(pts_est, k=1)
    est_to_gt_mean = est_to_gt_dists.mean()
    
    # Calculate distances from ground truth points to nearest estimated points
    gt_to_est_dists, _ = tree_est.query(pts_gt, k=1)
    gt_to_est_mean = gt_to_est_dists.mean()
    
    # Chamfer distance is the mean of both directional distances
    chamfer_dist = (est_to_gt_mean + gt_to_est_mean) / 2
    return chamfer_dist


# ============================================================================
# ADDITIONAL METRICS
# ============================================================================

def re(R_est, R_gt):
    """Rotational Error.
    
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error in degrees.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))
    
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, error_cos))
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg]
    return error


def te(t_est, t_gt):
    """Translational Error.
    
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error
