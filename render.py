#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render as render_3dgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp,
               separate_sh, render_fn, is_surfel, is_stage_b=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_package = render_fn(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = render_package["render"]
        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if is_stage_b:
            from utils.reflection_debug import save_reflection_debug_maps
            gbuffer_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gbuffer", "{:05d}".format(idx))
            save_reflection_debug_maps(
                render_package,
                gt,
                gbuffer_path,
                specular_mask=view.specular_mask,
                mask_sha256=view.specular_mask_sha256,
            )
        elif is_surfel:
            from utils.surfel_debug import save_surfel_debug_maps
            gbuffer_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gbuffer", "{:05d}".format(idx))
            save_surfel_debug_maps(render_package, gt, gbuffer_path)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        is_stage_b = getattr(dataset, "stage", "stage_a") == "stage_b"
        if is_stage_b:
            if dataset.model_type != "surfel":
                raise ValueError("Stage B render requires --model_type surfel")
            from gaussian_renderer.reflection_renderer import StageBRenderState, render as render_fn
            from scene.diffuse_surfel_model import DiffuseSurfelModel
            from scene.reflection_surfel_model import ReflectionSurfelModel
            from scene.stage_b_scene import StageBScene
            diffuse = DiffuseSurfelModel(dataset.roughness_min)
            reflection = ReflectionSurfelModel()
            scene = StageBScene(
                dataset, diffuse, reflection, load_iteration=iteration, shuffle=False, write_metadata=False
            )
            gaussians = StageBRenderState(
                diffuse=diffuse,
                reflection=reflection,
                scene_radius=scene.cameras_extent,
                ray_chunk_size=dataset.ray_chunk_size,
                ray_cutoff_sigma=dataset.ray_cutoff_sigma,
                ray_hit_threshold=dataset.ray_hit_threshold,
                ray_epsilon_scale=dataset.ray_epsilon_scale,
                material_alpha_threshold=dataset.material_alpha_threshold,
                roughness_min=diffuse.roughness_min,
                roughness_remap=dataset.roughness_remap,
                ray_background=dataset.ray_background,
            )
            is_surfel = False
        elif dataset.model_type == "surfel":
            from gaussian_renderer.surfel_renderer import render as render_fn
            from scene.diffuse_surfel_model import DiffuseSurfelModel
            gaussians = DiffuseSurfelModel(dataset.roughness_min)
            is_surfel = True
        elif dataset.model_type == "3dgs":
            gaussians = GaussianModel(dataset.sh_degree)
            render_fn = render_3dgs
            is_surfel = False
        else:
            raise ValueError("--model_type must be either '3dgs' or 'surfel'")
        if not is_stage_b:
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, render_fn, is_surfel, is_stage_b)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, render_fn, is_surfel, is_stage_b)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)
