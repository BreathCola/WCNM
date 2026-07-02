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

import os
import math
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, normal_depth_consistency_loss, monocular_normal_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.stage_a_state import make_full_stage_a_checkpoint, restore_full_stage_a_checkpoint
from utils.d_bootstrap_telemetry import (
    DBootstrapTelemetryWriter,
    SCHEMA_NAME as D_TELEMETRY_SCHEMA_NAME,
    SCHEMA_VERSION as D_TELEMETRY_SCHEMA_VERSION,
    validate_d_telemetry_options,
)
from utils.stage_b_telemetry import cuda_allocator_snapshot
from utils.training_state import (
    capture_rng_state,
    make_camera_runtime_state,
    restore_camera_deck,
    restore_rng_state,
    should_step_optimizer,
    validate_tier2_experiment_identity,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _stage_a_checkpoint_config(dataset, opt):
    return {
        "stage": getattr(dataset, "stage", "stage_a"),
        "model_type": dataset.model_type,
        "source_path": dataset.source_path,
        "images": dataset.images,
        "experiment": getattr(dataset, "experiment", ""),
        "resolution": int(dataset.resolution),
        "white_background": bool(dataset.white_background),
        "roughness_min": float(dataset.roughness_min),
        "normal_priors": dataset.normal_priors,
        "normal_prior_space": dataset.normal_prior_space,
        "optimizer_type": opt.optimizer_type,
        "random_background": bool(opt.random_background),
        "lambda_dssim": float(opt.lambda_dssim),
        "lambda_norm": float(opt.lambda_norm),
        "lambda_mono": float(opt.lambda_mono),
        "lambda_perc": float(opt.lambda_perc),
        "position_lr_init": float(opt.position_lr_init),
        "position_lr_final": float(opt.position_lr_final),
        "position_lr_delay_mult": float(opt.position_lr_delay_mult),
        "position_lr_max_steps": int(opt.position_lr_max_steps),
        "feature_lr": float(opt.feature_lr),
        "material_lr": float(opt.material_lr),
        "opacity_lr": float(opt.opacity_lr),
        "scaling_lr": float(opt.scaling_lr),
        "rotation_lr": float(opt.rotation_lr),
        "exposure_lr_init": float(opt.exposure_lr_init),
        "exposure_lr_final": float(opt.exposure_lr_final),
        "exposure_lr_delay_steps": int(opt.exposure_lr_delay_steps),
        "exposure_lr_delay_mult": float(opt.exposure_lr_delay_mult),
        "percent_dense": float(opt.percent_dense),
        "densify_from_iter": int(opt.densify_from_iter),
        "densify_until_iter": int(opt.densify_until_iter),
        "densification_interval": int(opt.densification_interval),
        "densify_grad_threshold": float(opt.densify_grad_threshold),
        "opacity_reset_interval": int(opt.opacity_reset_interval),
        "operator_gate_continuation": bool(opt.operator_gate_continuation),
    }


def _validate_stage_a_resume_config(saved, current):
    for key, value in current.items():
        if saved.get(key) != value:
            raise ValueError(f"full-state Stage A resume config mismatch for {key}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             diffuse_init_checkpoint=None):

    if getattr(dataset, "stage", "stage_a") == "stage_b":
        from stage_b_training import training_stage_b
        return training_stage_b(
            dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
            checkpoint, diffuse_init_checkpoint, debug_from, prepare_output_and_logger,
        )

    if dataset.model_type not in ("3dgs", "surfel"):
        raise ValueError("--model_type must be either '3dgs' or 'surfel'")
    is_surfel = dataset.model_type == "surfel"
    d_telemetry_enabled = validate_d_telemetry_options(
        getattr(opt, "d_bootstrap_telemetry_jsonl", ""),
        getattr(opt, "d_bootstrap_telemetry_max_steps", 0),
        getattr(opt, "d_bootstrap_telemetry_phase_tag", ""),
    )
    if getattr(opt, "operator_gate_continuation", False):
        validate_tier2_experiment_identity(dataset.experiment, dataset.resolution)
    if is_surfel:
        if dataset.normal_prior_space not in ("camera", "world"):
            raise ValueError("--normal_prior_space must be either 'camera' or 'world'")
        if opt.debug_interval <= 0:
            raise ValueError("--debug_interval must be positive")
        from gaussian_renderer.surfel_renderer import render as render_fn
        from scene.diffuse_surfel_model import DiffuseSurfelModel
        gaussians = DiffuseSurfelModel(dataset.roughness_min, opt.optimizer_type)
        if getattr(opt, "operator_gate_continuation", False) and not d_telemetry_enabled:
            raise ValueError("operator-gated D continuation requires bounded D bootstrap telemetry")
    elif d_telemetry_enabled or getattr(opt, "operator_gate_continuation", False):
        raise ValueError("D bootstrap telemetry/continuation is surfel-only")
    else:
        render_fn = render
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    if not is_surfel and not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    runtime_state = None
    operator_rng_state = None
    if checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location="cuda" if is_surfel else None)
        if is_surfel:
            if not isinstance(checkpoint_data, dict) or checkpoint_data.get("format") != "rtgs_stage_a":
                raise ValueError("surfel mode requires an RT-GS Stage A checkpoint")
            if opt.operator_gate_continuation:
                first_iter, runtime_state, saved_config = restore_full_stage_a_checkpoint(
                    checkpoint_data, gaussians, opt, restore_rng=True
                )
                operator_rng_state = capture_rng_state()
                _validate_stage_a_resume_config(saved_config, _stage_a_checkpoint_config(dataset, opt))
                model_params = None
            else:
                model_params = checkpoint_data["model_state"]
                first_iter = checkpoint_data["iteration"]
        else:
            model_params, first_iter = checkpoint_data
        if model_params is not None:
            gaussians.restore(model_params, opt)

    if opt.operator_gate_continuation:
        expected_start = int(opt.operator_gate_expected_global_start)
        if expected_start < 0 or first_iter != expected_start:
            raise ValueError(
                f"operator gate expected global start {expected_start}, restored {first_iter}"
            )
        if opt.iterations <= first_iter:
            raise ValueError("operator gate endpoint must be greater than its restored start")

    d_telemetry = None
    if d_telemetry_enabled:
        d_telemetry = DBootstrapTelemetryWriter(
            opt.d_bootstrap_telemetry_jsonl,
            opt.d_bootstrap_telemetry_max_steps,
            opt.d_bootstrap_telemetry_phase_tag,
        )
        d_telemetry.validate_planned_steps(opt.iterations - first_iter)

    perceptual_loss_fn = None
    if is_surfel and opt.lambda_perc > 0:
        from utils.perceptual_loss import VGG16PerceptualLoss
        perceptual_loss_fn = VGG16PerceptualLoss(pretrained=True).cuda().eval()
    if operator_rng_state is not None:
        # Model/helper construction during process restart is not part of the
        # continuous training trajectory.
        restore_rng_state(operator_rng_state)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = not is_surfel and opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    all_train_cameras = scene.getTrainCameras().copy()
    viewpoint_stack, viewpoint_indices = restore_camera_deck(all_train_cameras, runtime_state)
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    require_nonzero_mono = is_surfel and opt.require_nonzero_mono
    mono_supervised_steps = 0
    mono_nonzero_steps = 0
    mono_max = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        d_wall_start = time.perf_counter() if d_telemetry is not None else None
        d_count_before = int(gaussians.get_xyz.shape[0]) if d_telemetry is not None else None
        d_topology_called = False
        if d_telemetry is not None:
            torch.cuda.reset_peak_memory_stats()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render_fn(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = all_train_cameras.copy()
            viewpoint_indices = list(range(len(all_train_cameras)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Lnorm = image.new_zeros(())
        Lmono = image.new_zeros(())
        Lperc = image.new_zeros(())
        if is_surfel:
            Lnorm = normal_depth_consistency_loss(
                render_pkg["normal"], render_pkg["position"], render_pkg["alpha"]
            )
            loss = loss + opt.lambda_norm * Lnorm

            if viewpoint_cam.normal_prior is not None:
                from utils.surfel_utils import camera_normals_to_world, face_forward
                normal_prior = viewpoint_cam.normal_prior.permute(1, 2, 0)
                if viewpoint_cam.normal_prior_space == "camera":
                    normal_prior = camera_normals_to_world(viewpoint_cam, normal_prior)
                elif viewpoint_cam.normal_prior_space != "world":
                    raise ValueError("normal_prior_space must be 'camera' or 'world'")
                normal_prior = face_forward(
                    normal_prior, render_pkg["position"], viewpoint_cam.camera_center
                )
                prior_valid = viewpoint_cam.normal_prior_valid.permute(1, 2, 0)
                Lmono = monocular_normal_loss(
                    render_pkg["normal"], normal_prior, render_pkg["alpha"], prior_valid
                )
                loss = loss + opt.lambda_mono * Lmono
                if require_nonzero_mono:
                    mono_value = float(Lmono.detach().item())
                    mono_supervised_steps += 1
                    if torch.isfinite(Lmono.detach()).item() and mono_value > 0.0:
                        mono_nonzero_steps += 1
                        mono_max = max(mono_max, mono_value)
                        if mono_nonzero_steps == 1:
                            print(
                                "\n[MONO PRIOR] iteration={} view={} L_mono={:.8f}".format(
                                    iteration, viewpoint_cam.image_name, mono_value
                                )
                            )

            if perceptual_loss_fn is not None:
                Lperc = perceptual_loss_fn(image, gt_image)
                loss = loss + opt.lambda_perc * Lperc

        # Depth regularization
        Ll1depth_pure = 0.0
        if not is_surfel and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_value = float(loss.item())
            ema_loss_for_log = 0.4 * loss_value + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer and is_surfel:
                tb_writer.add_scalar('train_loss_patches/normal_depth', Lnorm.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/monocular_normal', Lmono.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/perceptual', Lperc.item(), iteration)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fn, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if is_surfel and (iteration % opt.debug_interval == 0 or iteration == opt.iterations):
                from utils.surfel_debug import save_surfel_debug_maps
                fixed_views = scene.getTestCameras() or scene.getTrainCameras()
                fixed_view = fixed_views[0]
                debug_package = render_fn(
                    fixed_view, gaussians, pipe, background,
                    use_trained_exp=dataset.train_test_exp,
                )
                debug_directory = os.path.join(scene.model_path, "debug", "iteration_{:06d}".format(iteration))
                save_surfel_debug_maps(debug_package, fixed_view.original_image.cuda(), debug_directory)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    d_topology_called = True
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            optimizer_step_completed = should_step_optimizer(
                iteration, opt.iterations, opt.operator_gate_continuation
            )
            if optimizer_step_completed:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                if is_surfel:
                    if opt.operator_gate_continuation:
                        checkpoint_data = make_full_stage_a_checkpoint(
                            gaussians,
                            iteration,
                            _stage_a_checkpoint_config(dataset, opt),
                            make_camera_runtime_state(viewpoint_indices, len(all_train_cameras)),
                            optimizer_step_completed,
                        )
                    else:
                        checkpoint_data = {
                            "format": "rtgs_stage_a",
                            "iteration": iteration,
                            "model_state": gaussians.capture(),
                            "config": {
                                "model_type": dataset.model_type,
                                "roughness_min": dataset.roughness_min,
                                "lambda_norm": opt.lambda_norm,
                                "lambda_mono": opt.lambda_mono,
                                "lambda_perc": opt.lambda_perc,
                            },
                        }
                    torch.save(checkpoint_data, checkpoint_path)
                else:
                    torch.save((gaussians.capture(), iteration), checkpoint_path)

            if d_telemetry is not None:
                whole_step_wall_ms = float((time.perf_counter() - d_wall_start) * 1000.0)
                d_count = int(gaussians.get_xyz.shape[0])
                allocator = cuda_allocator_snapshot(
                    True, f"since_explicit_d_telemetry_step_start_reset_at_global_{iteration}"
                )
                d_telemetry.append({
                    "schema_name": D_TELEMETRY_SCHEMA_NAME,
                    "schema_version": D_TELEMETRY_SCHEMA_VERSION,
                    "telemetry_step": d_telemetry.count + 1,
                    "global_iteration": int(iteration),
                    "camera_stem": str(viewpoint_cam.image_name),
                    "phase_tag": d_telemetry.phase_tag,
                    "total_loss": loss_value,
                    "d_count": d_count,
                    "d_count_delta": d_count - d_count_before,
                    "d_topology_event": bool(d_topology_called or d_count != d_count_before),
                    "whole_step_wall_ms": whole_step_wall_ms,
                    "whole_step_wall_definition": (
                        "CPU perf_counter from loop entry through normal debug/save/checkpoint and "
                        "optimizer/densification; excludes telemetry serialization; no added CUDA synchronize"
                    ),
                    **allocator,
                    "nonfinite_count": int(not math.isfinite(loss_value)),
                    "unavailable_fields": [],
                })

    if require_nonzero_mono:
        print(
            "\n[MONO PRIOR SUMMARY] supervised_steps={} nonzero_steps={} max_L_mono={:.8f}".format(
                mono_supervised_steps, mono_nonzero_steps, mono_max
            )
        )
        if mono_nonzero_steps == 0:
            raise RuntimeError(
                "--require_nonzero_mono was set, but no positive finite L_mono was observed"
            )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    rendered_package = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(rendered_package["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if getattr(scene.gaussians, "model_type", None) == "surfel":
                            tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), rendered_package["alpha"].permute(2, 0, 1)[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), (rendered_package["normal"].permute(2, 0, 1)[None] * 0.5 + 0.5).clamp(0, 1), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--diffuse_init_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.diffuse_init_checkpoint)

    # All done
    print("\nTraining complete.")
