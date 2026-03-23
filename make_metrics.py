import numpy as np
import json
from datareader import Ho3dReader
import math
from sklearn.neighbors import KDTree
import argparse
import os
import trimesh
from scipy.spatial import cKDTree
from Utils import compute_auc_sklearn
from pose_metrics import calculate_3d_iou, transform_pts_Rt, decompose_pose_matrix

def pose_to_Rt(pose):
    R = pose[:3, :3]
    t = pose[:3, 3]
    return R, t

def calc_pts_diameter(pts):
  """Calculates the diameter of a set of 3D points (i.e. the maximum distance
  between any two points in the set).

  :param pts: nx3 ndarray with 3D points.
  :return: The calculated diameter.
  """
  diameter = -1.0
  for pt_id in range(pts.shape[0]):
    pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
    pts_diff = pt_dup - pts[pt_id:, :]
    max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
    if max_dist > diameter:
      diameter = max_dist
  return diameter

def np_transform_pcd(pcd: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd

def compute_add(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pose_to_Rt(pred_pose)
    gt_r, gt_t = pose_to_Rt(gt_pose)

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD computation
    add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

    return add

def compute_adds(pcd: np.ndarray, pred_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    pred_r, pred_t = pose_to_Rt(pred_pose)
    gt_r, gt_t = pose_to_Rt(gt_pose)

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    # ADD-S computation
    kdt = KDTree(model_gt, metric='euclidean')
    distance, _ = kdt.query(model_pred, k=1)
    adds = np.mean(distance)

    return adds

def compute_cd(pred_pcd: np.ndarray, gt_pcd, pred_pose: np.ndarray, gt_pose: np.ndarray) -> float:
    pred_r, pred_t = pose_to_Rt(pred_pose)
    gt_r, gt_t = pose_to_Rt(gt_pose)

    pred_pts = np_transform_pcd(pred_pcd, pred_r, pred_t)
    gt_pts = np_transform_pcd(gt_pcd, gt_r, gt_t)
    
    # Create KD-trees for efficient nearest neighbor search
    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)
    
    # Calculate distances from estimated points to nearest ground truth points
    est_to_gt_dists, _ = gt_tree.query(pred_pts, k=1)
    est_to_gt_mean = est_to_gt_dists.mean()
    
    # Calculate distances from ground truth points to nearest estimated points
    gt_to_est_dists, _ = pred_tree.query(gt_pts, k=1)
    gt_to_est_mean = gt_to_est_dists.mean()
    
    # Chamfer distance is the mean of both directional distances
    chamfer_dist = (est_to_gt_mean + gt_to_est_mean) / 2
    return float(chamfer_dist)

def load_est_pose(i_str, debug_dir):
    return np.loadtxt(f'{debug_dir}/ob_in_cam/{i_str}.txt').reshape(4, 4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', type=str, default='SM1')
    parser.add_argument('--debug_path', type=str, default='../My-UA-Pose')#'/Experiments/simonep01/Results')
    parser.add_argument('--method', type=str, default='ua')    
    args = parser.parse_args()

    debug_dir = f'{args.debug_path}/{args.method}/{args.video_id}'
    eval_dir = f'{debug_dir}/evaluation_results'

    test_scene_dir = f'/Experiments/simonep01/ho3d/evaluation/{args.video_id}'
    reader = Ho3dReader(test_scene_dir)
    gt_mesh = reader.get_gt_mesh()
    gt_diameter = calc_pts_diameter(np.array(gt_mesh.vertices))
    print(f'diameter is: {gt_diameter}')

    pred_pose_0 = load_est_pose(reader.id_strs[0], debug_dir)
    gt_pose_0 = reader.get_gt_pose(0)

    pred_mesh = trimesh.load(f'{debug_dir}/final_mesh.obj')        
    
    object_metrics = {
        'ADD': [],
        'ADDS': []
    }

    for i in range(len(reader.color_files)):
        gt_pose = reader.get_gt_pose(i)
        pred_pose_i = load_est_pose(reader.id_strs[i], debug_dir)
        pred_pose = pred_pose_i @ np.linalg.inv(pred_pose_0) @ gt_pose_0

        add = compute_add(gt_mesh.vertices, pred_pose, gt_pose)
        adds = compute_adds(gt_mesh.vertices, pred_pose, gt_pose)

        object_metrics['ADD'].append(float(add))
        object_metrics['ADDS'].append(float(adds))

    add_normalized = np.array(object_metrics['ADD']) / gt_diameter
    adds_normalized = np.array(object_metrics['ADDS']) / gt_diameter
    auc_add = compute_auc_sklearn(add_normalized, max_val=0.1)
    auc_adds = compute_auc_sklearn(adds_normalized, max_val=0.1)
    # Load 3D IoU and ADI from run_attachment.py output
    iou_file = os.path.join(eval_dir, '3D_IOU_per_frame.json')
    adi_file = os.path.join(eval_dir, 'ADI_per_frame.json')
    
    if os.path.exists(iou_file):
        with open(iou_file, 'r') as f:
            object_metrics['3D_IOU'] = json.load(f)
    
    if os.path.exists(adi_file):
        with open(adi_file, 'r') as f:
            object_metrics['ADI'] = json.load(f)

    # Save all per-frame metrics to metrics.json
    metrics_file = os.path.join(eval_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(object_metrics, f, indent=2)

    # Add statistics for all metrics
    summary={}
    for key, values in object_metrics.items():
        if len(values) > 0:
            if key in ['ADD', 'ADDS']:
                # Save per-frame mean/min/max as _val (in cm)
                summary[f'{key}_val'] = {
                    'mean': float(np.mean(values) * 100),
                    'min': float(np.min(values) * 100),
                    'max': float(np.max(values) * 100)
                }
            elif key == 'ADI':
                summary[key] = {
                    'mean': float(np.mean(values) * 100),
                    'min': float(np.min(values) * 100),
                    'max': float(np.max(values) * 100)
                }
            else:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

    # Save AUC as ADD and ADDS
    summary['ADD']=float(auc_add * 100)
    summary['ADDS']=float(auc_adds * 100)

    chamfer_distance = compute_cd(pred_mesh.vertices, gt_mesh.vertices, pred_pose_0, gt_pose_0)
    summary['CD']=float(chamfer_distance*100)  # m to cm

    # Compute scale: 3D IoU of final_mesh at pred_pose_0 vs gt_mesh at gt_pose_0
    R_est, t_est = decompose_pose_matrix(pred_pose_0)
    R_gt, t_gt = decompose_pose_matrix(gt_pose_0)
    pts_est = transform_pts_Rt(np.array(pred_mesh.vertices), R_est, t_est)
    pts_gt = transform_pts_Rt(np.array(gt_mesh.vertices), R_gt, t_gt)
    summary['scale'] = float(calculate_3d_iou(pts_est, pts_gt))
    
    # Save updated summary
    summary_file = os.path.join(eval_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved per-frame metrics to {metrics_file}")
    print(f"Updated summary saved to {summary_file}")

    # Print results
    print(f"{'='*60}")
    print(f"Evaluation Results for video {args.video_id} with method {args.method}")
    print(f"{'='*60}")
    if 'ADI' in summary:
        print(f"ADI (Average Distance):        {summary['ADI']['mean']:.4f} cm")
    if 'ADD_val' in summary:
        print(f"ADD (mean):                    {summary['ADD_val']['mean']:.2f} cm")
    if 'ADDS_val' in summary:
        print(f"ADDS (mean):                   {summary['ADDS_val']['mean']:.2f} cm")
    if 'CD' in summary:
        print(f"Chamfer distance:              {summary['CD']:.2f} cm")
    if '3D_IOU' in summary:
        print(f"3D IOU:                        {summary['3D_IOU']['mean']:.3f} %")
    if 'ADD' in summary:
        print(f"ADD (AUC):                     {summary['ADD']:.2f} %")
    if 'ADDS' in summary:
        print(f"ADDS (AUC):                    {summary['ADDS']:.2f} %")
    if 'scale' in summary:
        print(f"Scale (3D IoU):                {summary['scale']:.3f} %")
    print(f"{'='*60}\n")
