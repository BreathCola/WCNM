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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.model_type = "3dgs"
        self.stage = "stage_a"
        self.experiment = ""
        self.roughness_min = 0.03
        self.normal_priors = "normal_priors"
        self.normal_prior_space = "camera"
        self.specular_masks = ""
        self.internal_object_masks = ""
        self.reflection_init_mode = "random_bbox"
        self.reflection_init_count = 0
        self.reflection_init_seed = 0
        self.geometry_release_manifest = ""
        self.transmittance_init_mode = "random_bbox"
        self.transmittance_init_count = 4096
        self.transmittance_init_seed = 0
        self.transmittance_compose = "alpha_over"
        self.transparent_path_mode = "legacy_d_gbuffer"
        self.transparent_direct_mode = "legacy"
        self.transparent_reflection_mode = "legacy"
        self.cout_ownership_mode = "legacy"
        self.stage_d_static_cache_path = ""
        self.ray_background = "scene"
        self.ray_chunk_size = 4096
        self.ray_cutoff_sigma = 3.0
        self.ray_hit_threshold = 0.0001
        self.ray_epsilon_scale = 0.0001
        self.material_alpha_threshold = 0.0001
        self.roughness_remap = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.material_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_norm = 0.04
        self.lambda_mono = 0.01
        self.lambda_perc = 0.01
        self.require_nonzero_mono = False
        self.lambda_spec = 0.2
        self.specular_k0 = 0.9
        self.specular_smoke_diagnostics = False
        self.stage_b_telemetry_jsonl = ""
        self.stage_b_telemetry_max_steps = 0
        self.stage_b_telemetry_phase_tag = ""
        self.d_bootstrap_telemetry_jsonl = ""
        self.d_bootstrap_telemetry_max_steps = 0
        self.d_bootstrap_telemetry_phase_tag = ""
        self.operator_gate_continuation = False
        self.operator_gate_expected_global_start = -1
        self.operator_gate_expected_r_local_start = -1
        self.tier2_retry_identity = ""
        self.stage_b_allocator_policy = "default"
        self.stage_b_reference_peak_allocated_bytes = 0
        self.stage_b_minimum_projected_headroom_bytes = 0
        self.stage_b_pressure_release_free_bytes = 0
        self.stage_b_memory_retry_min_chunk_size = 0
        self.debug_interval = 1000
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.reflection_position_lr_init = 0.00016
        self.reflection_position_lr_final = 0.0000016
        self.reflection_position_lr_delay_mult = 0.01
        self.reflection_position_lr_max_steps = 20_000
        self.reflection_color_lr = 0.0025
        self.reflection_opacity_lr = 0.025
        self.reflection_scaling_lr = 0.005
        self.reflection_rotation_lr = 0.001
        self.reflection_percent_dense = 0.01
        self.reflection_densify_from_iter = 100
        self.reflection_densify_until_iter = 5_000
        self.reflection_densification_interval = 100
        self.reflection_densify_grad_threshold = 0.0002
        self.reflection_min_opacity = 0.005
        self.reflection_prune_unhit_after = 500
        self.lambda_depth = 0.2
        self.stage_d_depth_start_iteration = 40_000
        self.stage_d_smoke = False
        self.stage_d_smoke_max_steps = 3
        self.stage_d_formal_onset = False
        self.stage_d_cached_twarmup = False
        self.stage_d_semantic_repair_pilot = False
        self.stage_d_ownership_pilot = False
        self.stage_d_ownership_arm = ""
        self.stage_d_reuse_static_cache = False
        self.stage_d_ownership_t_long = False
        self.stage_d_tscale_recovery_preflight = False
        self.stage_d_tscale_recovery_long = False
        self.stage_d_internal_object_pilot = False
        self.stage_d_internal_object_to_20000 = False
        self.stage_d_phase_a_end_iteration = 18_000
        self.stage_d_cache_parity_atol = 2e-5
        self.stage_d_cache_parity_mean_atol = 2e-6
        self.transparent_interface_margin = 0.05
        self.transparent_interface_margin_mode = "exclude"
        self.transfer_min_views = 3
        self.transfer_min_total_weight = 0.01
        self.transfer_depth_margin = 0.05
        self.transfer_opacity_scale = 0.25
        self.transfer_opacity_min = 0.005
        self.transfer_opacity_max = 0.05
        self.object_mask_min_views = 3
        self.object_mask_min_support_ratio = 0.6
        self.object_mask_boundary_ignore_px = 5
        self.object_mask_init_mode = "reviewed_union_v1"
        self.object_mask_erode_px = 3
        self.object_mask_dilate_px = 3
        self.object_alpha_floor = 0.35
        self.lambda_object_positive = 0.0
        self.lambda_object_negative = 0.0
        self.lambda_anti_veil_black = 0.05
        self.lambda_anti_veil_saturation = 0.02
        self.anti_veil_gt_luminance_threshold = 0.15
        self.anti_veil_high_alpha_threshold = 0.80
        self.anti_veil_black_luminance_threshold = 0.08
        self.anti_veil_saturation_alpha_threshold = 0.95
        self.anti_veil_target_saturation_coverage = 0.35
        self.anti_veil_gate_temperature = 0.05
        self.anti_veil_black_temperature = 0.02
        self.anti_veil_coverage_temperature = 0.02
        self.anti_veil_epsilon = 1e-6
        self.anti_veil_ramp_start = 0
        self.anti_veil_ramp_end = 200
        self.stage_d_telemetry_jsonl = ""
        self.transmittance_position_lr_init = 0.00016
        self.transmittance_position_lr_final = 0.0000016
        self.transmittance_position_lr_delay_mult = 0.01
        self.transmittance_position_lr_max_steps = 20_000
        self.transmittance_color_lr = 0.0025
        self.transmittance_opacity_lr = 0.025
        self.transmittance_scaling_lr = 0.005
        self.transmittance_rotation_lr = 0.001
        self.transmittance_percent_dense = 0.01
        self.transmittance_densify_from_iter = 100
        self.transmittance_densify_until_iter = 5_000
        self.transmittance_densification_interval = 100
        self.transmittance_densify_grad_threshold = 0.0002
        self.transmittance_min_opacity = 0.005
        self.transmittance_prune_unhit_after = 500
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
