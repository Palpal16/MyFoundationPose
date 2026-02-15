import torch
import trimesh
import open3d as o3d
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix, Transform3d

#videos = ['AP11', 'AP12', 'AP13', 'MPM11', 'MPM12', 'MPM13', 'MPM14']

videos=['AP10']

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T
my_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)

def compose_transform(scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> Transform3d:
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)

def downsample_mesh_best_quality(mesh, target_vertices):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    colors = mesh.visual.vertex_colors[:, :3] / 255.0
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_o3d.compute_vertex_normals()
    target_triangles = target_vertices * 2
    mesh_simplified = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    vertices_out = np.asarray(mesh_simplified.vertices)
    faces_out = np.asarray(mesh_simplified.triangles)
    colors_out = (np.asarray(mesh_simplified.vertex_colors) * 255).astype(np.uint8)
    mesh_out = trimesh.Trimesh(vertices=vertices_out, faces=faces_out, vertex_colors=colors_out)
    mask = mesh_out.nondegenerate_faces()
    mesh_out.update_faces(mask)
    mesh_out.remove_infinite_values()
    trimesh.repair.fix_normals(mesh_out)
    return mesh_out

for video_id in videos:
    print(f'\n{"="*60}')
    print(f"Processing video: {video_id}")
    print(f'{"="*60}\n')
    sam_dir = f'/home/simonep01/sam-3d-objects/meshes/{video_id}'
    output = torch.load(f'{sam_dir}/output_data.pt', map_location='cuda:0')

    mesh = output['glb'].copy()
    mesh.export(f'{sam_dir}/initial_mesh.obj')

    vertices = mesh.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
    vertices_tensor = torch.from_numpy(vertices).float().to(output["rotation"].device)
    R_l2c = quaternion_to_matrix(output['rotation'])
    l2c_transform = compose_transform(scale=output['scale'], rotation=R_l2c, translation=output['translation'])
    vertices_transformed = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
    mesh.vertices = vertices_transformed.squeeze(0).cpu().numpy() @ my_rotation

    mesh.export(f'{sam_dir}/transformed_mesh.obj')

    mesh_simplified = downsample_mesh_best_quality(mesh=mesh, target_vertices=60000)
    mesh_simplified.export(f'{sam_dir}/reduced_mesh.obj')
