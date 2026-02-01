# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,os,sys

class Ho3dReader:
  def __init__(self,video_dir, root_dir='/Experiments/simonep01/ho3d',downscale=1):
    self.video_dir = video_dir
    self.downscale = downscale
    self.root_dir = root_dir
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
    meta_file = self.color_files[0].replace('.jpg','.pkl').replace('rgb','meta')
    self.K = pickle.load(open(meta_file,'rb'))['camMat']
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_poses = np.loadtxt(f'{self.video_dir}/ob_in_cams.txt').reshape(-1,4,4)
    
    self.id_strs = []
    for i in range(len(self.color_files)):
      id = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(id)

  def __len__(self):
    return len(self.color_files)

  def get_video_name(self):
    return os.path.dirname(os.path.abspath(self.color_files[0])).split('/')[-2]

  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i,use_gt_mask=False,erode=False,xmem=False):
    print(self.color_files[i].replace('.jpg','.png').replace('evaluation','masks_XMem').replace('rgb/',''))
    mask = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('evaluation','masks_XMem').replace('rgb/',''), -1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    ## ??? The ?hand? mask is sometimes set to 127 and reading with INTER_NEAREST, sees it as the object
    mask[mask<150]=0
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    if erode:
      kernel = np.ones((3, 3), np.uint8)
      mask = cv2.dilate(mask, kernel, iterations=1)
      mask = cv2.erode(mask, kernel, iterations=2)
    return mask

  def get_occ_mask(self,i):
    video_name = self.get_video_name()
    index = int(os.path.basename(self.color_files[i]).split('.')[0])
    mask = cv2.imread(f'{self.root_dir}/masks_XMem/{video_name}_hand/{index:04d}.png',-1)
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return mask
    
  def get_ob_name(self):
    video2name = {
      'AP': '019_pitcher_base',
      'MPM': '010_potted_meat_can',
      'SB': '021_bleach_cleanser',
      'SM': '006_mustard_bottle',
    }
    video_name = self.get_video_name()
    for k in video2name:
      if video_name.startswith(k):
        ob_name = video2name[k]
        break
    return ob_name
  
  def get_ob_id(self):
    obj_list = {
      '019_pitcher_base': '11',
      '010_potted_meat_can': '9',
      '021_bleach_cleanser': '12',
      '006_mustard_bottle': '5',
      }
    ob_name = self.get_ob_name()
    return obj_list[ob_name]

  def get_gt_mesh(self):
    assert self.root_dir is not None or os.path.exists(self.root_dir), "Not found path for GT meshes!"
    ob_name = self.get_ob_name()
    mesh = trimesh.load(f'{self.root_dir}/models/{ob_name}/textured_simple.obj')
    return mesh

  def get_depth(self,i):
    color = imageio.imread(self.color_files[i])
    depth_scale = 0.00012498664727900177
    depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = (depth[...,2]+depth[...,1]*256)*depth_scale
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth,self.K)
    return xyz_map

  def get_gt_pose(self,i):
    return self.gt_poses[i]
