# import os
# import open3d as o3d
# import trimesh

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import sys
def main(path):
    app = gui.Application.instance
    app.initialize()

    model = o3d.io.read_triangle_model(path)

    vis = o3d.visualization.O3DVisualizer("Textures", 1024, 768);
    for mi in model.meshes:
        vis.add_geometry(mi.mesh_name, mi.mesh,
                         model.materials[mi.material_idx]);
    vis.reset_camera_to_default()
    vis.DEPTH
    app.add_window(vis);
    app.instance.run()
main('D:/IDM/obj_models/train/03211117/4a21927379f965a9e4b68d3b17c43658/model.obj')
# os.path.exists('./mesh/model.obj')
# mesh = o3d.io.read_triangle_model('D:/IDM/obj_models/train/03211117/4a21927379f965a9e4b68d3b17c43658/model.obj', True)
# mesh = trimesh.load_mesh('D:/IDM/obj_models/train/03211117/4a21927379f965a9e4b68d3b17c43658/model.obj')
# o3d.io.write_triangle_mesh
# mesh = o3d.io.read_triangle_mesh('G:/project/label_script/aaaaa.obj', True, True )

# mesh: o3d.geometry.TriangleMesh
# mesh.compute_triangle_normals()
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh)
# vis.run()
# vis.destroy_window()