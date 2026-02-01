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
from run_attachment import estimate_and_scale_mesh

from pose_metrics import adi_est

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
  #parser.add_argument('--mesh_file', type=str, default='/Experiments/simonep01/ho3d/instantmesh_models/019_pitcher_base/mesh.obj')
  #parser.add_argument('--mesh_file', type=str, default='/Experiments/simonep01/ho3d/models/019_pitcher_base/textured_simple.obj')
  parser.add_argument('--video_id', type=str, default='SM1')
  parser.add_argument('--scene_root', type=str, default='/Experiments/simonep01/ho3d/evaluation')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=2)
  parser.add_argument('--debug_path', type=str, default='/home/simonep01/FoundationPose/debug')
  parser.add_argument('--method', type=str, default='cheat_any6d')
  parser.add_argument('--evaluation', type=bool, default=True)
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  test_scene_dir = f'{args.scene_root}/{args.video_id}'
  debug_dir = f'{args.debug_path}/{args.method}/{args.video_id}'
  
  debug = args.debug
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  reader = Ho3dReader(video_dir=test_scene_dir)
  if args.method == 'fp':
    mesh = reader.get_gt_mesh()
  elif args.method == 'any6d':
    ob_name = reader.get_ob_name()
    mesh_file = f'/Experiments/simonep01/any6d_meshes/{ob_name}/final_mesh_{ob_name}.obj'
    mesh = trimesh.load(mesh_file)
  elif args.method == 'cheat_any6d':
    ob_name = reader.get_ob_name()
    mesh_file = f'/Experiments/simonep01/any6d_meshes/{ob_name}/final_mesh_{ob_name}.obj'
    mesh = trimesh.load(mesh_file)
    mesh, _ = estimate_and_scale_mesh(mesh,reader, cheating_scale=True)
  elif args.method == 'sam3d':
    mesh_file = f'/home/simonep01/sam-3d-objects/meshes/{args.video_id}/reduced_mesh.obj'
    mesh = trimesh.load(mesh_file)
    mesh, _ = estimate_and_scale_mesh(mesh,reader, cheating_scale=True)
  elif args.method == 'full_sam3d':
    mesh_file = f'/home/simonep01/sam-3d-objects/meshes/{args.video_id}/transformed_mesh.obj'
    mesh = trimesh.load(mesh_file)
    mesh, _ = estimate_and_scale_mesh(mesh,reader, cheating_scale=True)

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  if args.evaluation:
    eval_dir = f"{debug_dir}/evaluation_results"
    os.makedirs(eval_dir, exist_ok=True)
    gt_mesh = reader.get_gt_mesh()
    metrics_keys = ['ADI', '3D_IOU']
    per_frame_metrics = {key: [] for key in metrics_keys}

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
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      ## cv2.imshow('1', vis[...,::-1])
      ## cv2.waitKey(1)

    if debug>=2 and i%20==0:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    if args.evaluation:
      frame_metrics = evaluate_frame(gt_mesh, reader.get_gt_pose(i), mesh, pose)
      for key in metrics_keys:
        per_frame_metrics[key].append(frame_metrics[key])

  mesh.export(f'{debug_dir}/final_mesh.obj')

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
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"ADI (Average Distance):        {summary['ADI']['mean']:.4f} mm")
    print(f"3D IOU:                        {summary['3D_IOU']['mean']:.3f} %")