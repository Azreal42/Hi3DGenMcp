import os
from typing import Optional
from PIL import Image
import numpy as np
import torch
from src.trellis.representations.mesh.cube2mesh import MeshExtractResult
from src.trellis.utils import postprocessing_utils
from trellis.pipelines import TrellisImageTo3DPipeline
import trimesh

os.environ['SPCONV_ALGO'] = 'native'

def to_trimesh(mesh: MeshExtractResult, transform_pose=False):
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertex_normals = mesh.comput_v_normals(mesh.vertices, mesh.faces)

    if transform_pose:
        transform_matrix = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
        vertices = vertices @ transform_matrix
        vertex_normals = vertex_normals.detach().cpu().numpy() @ transform_matrix
    else:
        vertex_normals = vertex_normals.detach().cpu().numpy()
    
    # Create the trimesh mesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=mesh.face_normal.detach().cpu().numpy(),
        vertex_normals=vertex_normals
    )
    
    return mesh

def main():
    normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')

        # Initialize pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("Stable-X/trellis-normal-v0-1")
    pipeline.cuda()

    # Initialize normal predictor
    

    image = Image.open(rf"D:\Dev\WolfRing.png")

    image = pipeline.preprocess_image(image, resolution=1024)
    normal_image = normal_predictor(image, resolution=768, match_input_resolution=True, data_type='object')

    ss_guidance_strength=3
    ss_sampling_steps=50
    slat_guidance_strength=3
    slat_sampling_steps=6
    seed=-1

    outputs = pipeline.run(
        normal_image,
        seed=seed,
        formats=["mesh",],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )

    generated_mesh = outputs['mesh'][0]

    mesh_path = rf"D:\Dev\WolfRing.glb"
    trimesh_mesh = to_trimesh(generated_mesh, transform_pose=True)
    trimesh_mesh.export(mesh_path)



if __name__ == "__main__":
    main()
