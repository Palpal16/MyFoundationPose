import numpy as np
import glob
import trimesh
import pickle
import os
import json
from pose_metrics import (
    calculate_3d_iou,
    calculate_3d_iou_with_pose,
    mssd_est,
    mspd_est,
    adi_est,
    chamfer_distance,
    mssd,
    mspd,
    adi,
    add
)



def pose_to_Rt(pose):
    """Convert 4x4 pose matrix to R and t.
    
    Args:
        pose: 4x4 transformation matrix
        
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    R = pose[:3, :3]
    t = pose[:3, 3:4]
    return R, t



def transform_pts_Rt(pts, R, t):
    """Transform points using R and t (mimics bop_toolkit_lib.misc.transform_pts_Rt)."""
    return (R @ pts.T + t).T



if __name__ == '__main__':
    # ========================================================================
    # Configuration
    # ========================================================================
    gt_mesh_file = '/Experiments/simonep01/demo_data/light_ho3d/models/019_pitcher_base/textured_simple.obj'
    est_mesh_file = gt_mesh_file
    pose_file = "/Experiments/simonep01/demo_data/light_ho3d/evaluation/AP14/ob_in_cams.txt"
    est_poses_dir = "debug/ob_in_cam/*.txt"
    meta_file = "/Experiments/simonep01/demo_data/light_ho3d/evaluation/AP14/meta/0000.pkl"
    
    # Flags
    use_gt_mesh = False  # Set to True to use efficient single-mesh functions
    debug = 0  # 0: normal, 1: print variance and save summary, 2: save per-frame results
    output_dir = 'debug/evaluation_results'  # Used when debug>=1
    
    # ========================================================================
    # Load camera, symmetries and poses
    # ========================================================================
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    
    K = meta['camMat']
    syms = [{"R": np.eye(3), "t": np.zeros((3, 1))}]


    gt_poses = np.loadtxt(pose_file).reshape(-1, 4, 4)
    est_pose_files = sorted(glob.glob(est_poses_dir))
    
    n_frames = min(len(est_pose_files), len(gt_poses))
    
    # ========================================================================
    # Load meshes
    # ========================================================================
    gt_mesh = trimesh.load(gt_mesh_file)
    pts_gt_orig = np.array(gt_mesh.vertices, dtype=np.float32)
    
    # Determine if we should use single-mesh (efficient) functions
    if use_gt_mesh:
        pts = pts_gt_orig
        print("Using single-mesh functions (GT mesh)")
    else:
        est_mesh = trimesh.load(est_mesh_file)
        pts_est_orig = np.array(est_mesh.vertices, dtype=np.float32)
        print("Using two-mesh functions (different meshes)")
    
    # ========================================================================
    # Initialize metric storage
    # ========================================================================
    metrics = {
        'ADD': 0.0,
        'ADI': 0.0,
        '3D_IOU': 0.0,
        'Chamfer': 0.0,
        'MSSD': 0.0,
        'MSPD': 0.0
    }
    
    # For variance calculation and per-frame storage
    if debug >= 1:
        per_frame_metrics = {key: [] for key in metrics.keys()}
    
    # Create output directory if debug>=1
    if debug >= 1:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # ========================================================================
    # Compute metrics for each frame
    # ========================================================================
    for i in range(n_frames):
        # Load estimated pose
        est_pose = np.loadtxt(est_pose_files[i]).reshape(4, 4)
        gt_pose = gt_poses[i]
        
        # Convert poses to R, t format
        R_est, t_est = pose_to_Rt(est_pose)
        R_gt, t_gt = pose_to_Rt(gt_pose)
        
        frame_metrics = {}
        
        if use_gt_mesh:
            # ADD metric
            add_error = add(R_est=R_est, t_est=t_est, R_gt=R_gt, t_gt=t_gt, pts=pts)
            frame_metrics['ADD'] = add_error
            
            # ADI metric
            adi_error = adi(R_est=R_est, t_est=t_est, R_gt=R_gt, t_gt=t_gt, pts=pts)
            frame_metrics['ADI'] = adi_error
            
            # 3D IOU
            iou = calculate_3d_iou_with_pose(R1=R_est, t1=t_est, R2=R_gt, t2=t_gt, pts=pts)
            frame_metrics['3D_IOU'] = iou
            
            # Chamfer distance (need to compute manually for single mesh)
            pts_est = transform_pts_Rt(pts, R_est, t_est)
            pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
            chamfer = chamfer_distance(R_est, t_est, pts, R_gt, t_gt, pts)
            frame_metrics['Chamfer'] = chamfer
            
            # MSSD metric
            mssd_val = mssd(R_est=R_est, t_est=t_est, R_gt=R_gt, t_gt=t_gt, pts=pts, syms=syms)
            frame_metrics['MSSD'] = mssd_val
            
            # MSPD metric
            mspd_val = mspd(R_est=R_est, t_est=t_est, R_gt=R_gt, t_gt=t_gt, K=K, pts=pts, syms=syms)
            frame_metrics['MSPD'] = mspd_val
            
        else:
            # ADD metric (compute manually)
            pts_est_t = transform_pts_Rt(pts_est_orig, R_est, t_est)
            pts_gt_t = transform_pts_Rt(pts_gt_orig, R_gt, t_gt)
            add_error = np.linalg.norm(pts_est_t - pts_gt_t, axis=1).mean()
            frame_metrics['ADD'] = add_error
            
            # ADI metric (returns IOU and error)
            iou, adi_error = adi_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig)
            frame_metrics['ADI'] = adi_error
            frame_metrics['3D_IOU'] = iou
            
            # Chamfer distance
            chamfer = chamfer_distance(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig)
            frame_metrics['Chamfer'] = chamfer
            
            # MSSD metric
            mssd_val = mssd_est(R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms)
            frame_metrics['MSSD'] = mssd_val
            
            # MSPD metric
            mspd_val = mspd_est(K, R_est, t_est, pts_est_orig, R_gt, t_gt, pts_gt_orig, syms)
            frame_metrics['MSPD'] = mspd_val
        
        # Accumulate metrics
        for key in metrics.keys():
            metrics[key] += frame_metrics[key]
            if debug >= 1:
                per_frame_metrics[key].append(frame_metrics[key])
    
    # ========================================================================
    # Compute average and standard deviation and save results
    # ========================================================================
    for key in metrics:
        metrics[key] /= n_frames
    
    if debug >= 1:
        std_devs = {}
        for key in metrics.keys():
            std_devs[key] = np.std(per_frame_metrics[key])

        # Save summary as JSON
        summary_data = {
            'num_frames': n_frames,
            'metrics': {}
        }
        
        for key in metrics.keys():
            summary_data['metrics'][key] = {
                'mean': float(metrics[key]),
                'std_dev': float(std_devs[key]),
                'min': float(np.min(per_frame_metrics[key])),
                'max': float(np.max(per_frame_metrics[key]))
            }
        
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to: {summary_file}")
    
    if debug == 2:
        # Save per-frame metrics as JSON
        for key in metrics.keys():
            per_frame_data = {
                'metric': key,
                'num_frames': n_frames,
                'values': [float(v) for v in per_frame_metrics[key]]
            }
            output_file = os.path.join(output_dir, f'{key}_per_frame.json')
            with open(output_file, 'w') as f:
                json.dump(per_frame_data, f, indent=2)
        print(f"Saved per-frame results to: {output_dir}")
    
    # ========================================================================
    # Print results
    # ========================================================================
    if debug == 0:
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({n_frames} frames)")
        print(f"{'='*60}")
        print(f"ADD (Average Distance):        {metrics['ADD']:.4f} mm")
        print(f"ADI (Average Distance):        {metrics['ADI']:.4f} mm")
        print(f"3D IOU:                        {metrics['3D_IOU']:.3f} %")
        print(f"Chamfer Distance:              {metrics['Chamfer']:.4f} mm")
        print(f"MSSD:                          {metrics['MSSD']:.4f} mm")
        print(f"MSPD:                          {metrics['MSPD']:.3f} pixels")
    else:
        print(f"\n{'='*60}")
        print("Variance Statistics")
        print(f"{'='*60}")
        print(f"ADD Variance:                  {metrics['ADD']:.6f} mm  ±{std_devs['ADD']:.4f}")
        print(f"ADI Variance:                  {metrics['ADI']:.6f} mm  ±{std_devs['ADI']:.4f}")
        print(f"3D IOU Variance:               {metrics['3D_IOU']:.5f} %  ±{std_devs['3D_IOU']:.4f}")
        print(f"Chamfer Distance Variance:     {metrics['Chamfer']:.6f} mm  ±{std_devs['Chamfer']:.4f}")
        print(f"MSSD Variance:                 {metrics['MSSD']:.6f} mm  ±{std_devs['MSSD']:.4f}")
        print(f"MSPD Variance:                 {metrics['MSPD']:.5f} pixels  ±{std_devs['MSPD']:.4f}")
    
    print(f"{'='*60}\n")