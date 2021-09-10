import trimesh
import open3d as o3d
import os
import numpy as np
from PIL import Image
from trimesh.visual import TextureVisuals


'''
Due to redundant uvs in open3d, the file size of saved by o3d.io.write_triangle_mesh is 
larger than the original one.

For example, the original file size if an '.obj' file with uvs is 20M, 
after saving by o3d.io.write_triangle_mesh (without any modification), the file size will 
become 31M (the additional 11M is the redundant uvs). 
'''



def save_mesh(save_path, mesh_o3d: o3d.geometry.TriangleMesh):
    mesh_trimesh = o3d_to_trimesh(mesh_o3d)
    name, suffix = os.path.splitext(os.path.basename(save_path))

    # 'material0.mtl' and 'material0.jpg' is used when trimesh saves a textured mesh as '.obj' file
    # Rewriting will occur if more than two '.obj' files save in a same dir.
    mesh_trimesh.visual.material.name = name
    return mesh_trimesh.export(save_path, include_normals=False)


def o3d_to_trimesh(mesh_o3d: o3d.geometry.TriangleMesh):
    # Convert o3d.geometry.TriangleMesh to trimesh.Trimesh
    # TODO: Add multiple textures support.
    assert mesh_o3d.has_triangles() and mesh_o3d.has_vertices()
    triangles = np.array(mesh_o3d.triangles)
    vertices = np.array(mesh_o3d.vertices)

    if mesh_o3d.has_triangle_uvs():
        uvs = np.array(mesh_o3d.triangle_uvs)
    if mesh_o3d.has_textures():
        img = np.array(mesh_o3d.textures[0])

    # remove the redundant uvs in o3d.geometry.TriangleMesh
    # assume that a o3d.geometry.TriangleMesh has triangle [0, 1, 2], [1, 2, 3]
    # 'uvs' will be recorded as [uvs0, uvs1, uvs2], [uvs1, uvs2, uvs3] in open3d
    # apparently, the 'uvs1' and 'uvs2' are recorded twice
    id_triangel2uvs = np.unique(triangles.flatten(), return_index=True)[1]
    uvs = uvs[id_triangel2uvs]

    img = Image.fromarray(img)
    # compared with original texture img, the texture img from open3d is vertically flipped.
    # therefore, the texture img from open3d needs to be converted to original texture img.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    visual = TextureVisuals(image=img, uv=uvs)
    mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=triangles, visual=visual)
    return mesh_trimesh