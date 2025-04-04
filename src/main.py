import os
from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from src.trellis.representations.mesh.cube2mesh import MeshExtractResult
from src.trellis.utils import postprocessing_utils
from trellis.pipelines import TrellisImageTo3DPipeline
import trimesh
from inference_ig2mv_sdxl import (
    prepare_pipeline,
    preprocess_image,
    remove_bg,
    run_pipeline,
)
from transformers import AutoModelForImageSegmentation
from torchvision import transforms

os.environ['SPCONV_ALGO'] = 'native'

def to_trimesh(mesh: MeshExtractResult, transform_pose=False) -> trimesh.Trimesh:
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

def texturate_mesh_with_mv_adapter(image_path:str, mesh_path:str):
    from huggingface_hub import hf_hub_download

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16
    MAX_SEED = np.iinfo(np.int32).max
    NUM_VIEWS = 6
    HEIGHT = 768
    WIDTH = 768
    TMP_DIR = os.path.expanduser

    pipe = prepare_pipeline(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        vae_model="madebyollin/sdxl-vae-fp16-fix",
        unet_model=None,
        lora_model=None,
        adapter_path="huanngzh/mv-adapter",
        scheduler=None,
        num_views=NUM_VIEWS,
        device=DEVICE,
        dtype=DTYPE,
    )
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to(DEVICE)
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, DEVICE)


    run_mvadapter(mesh_path, "", image_path)

    def run_mvadapter(
        mesh_path,
        prompt,
        image,
        seed=42,
        guidance_scale=3.0,
        num_inference_steps=30,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast"):
        # pre-process the reference image
        image = Image.open(image).convert("RGB") if isinstance(image, str) else image
        image = remove_bg_fn(image)
        image = preprocess_image(image, HEIGHT, WIDTH)

        if isinstance(seed, str):
            try:
                seed = int(seed.strip())
            except ValueError:
                seed = 42

        images, _, _, _ = run_pipeline(
            pipe,
            mesh_path=mesh_path,
            num_views=NUM_VIEWS,
            text=prompt,
            image=image,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            remove_bg_fn=None,
            reference_conditioning_scale=reference_conditioning_scale,
            negative_prompt=negative_prompt,
            device=DEVICE,
        )

        torch.cuda.empty_cache()

        return images, image
    

def generate_mesh_with_hi3dgen(image_path: str):
    normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1')

        # Initialize pipeline
    pipeline = TrellisImageTo3DPipeline.from_pretrained("Stable-X/trellis-normal-v0-1")
    pipeline.cuda()

    image = Image.open(image_path)

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

    torch.cuda.empty_cache()



if __name__ == "__main__":
    # generate_mesh_with_hi3dgen(rf"D:\Dev\WolfRing.png")
    texturate_mesh_with_mv_adapter(rf"D:\Dev\WolfRing.png", rf"D:\Dev\WolfRing.glb");
