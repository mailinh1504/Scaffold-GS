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
import subprocess

import numpy as np

cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
result = (
    subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split("\n")
)
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    np.argmin([int(x.split()[2]) for x in result[:-1]])
)

os.system("echo $CUDA_VISIBLE_DEVICES")


import json
import pathlib
import shutil
import sys
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import makedirs
from pathlib import Path
from random import randint

# from lpipsPyTorch import lpips
import lpips
import torch
import torchvision
import torchvision.transforms.functional as tf
import wandb
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, prefilter_voxel, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net="vgg").to("cuda")

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = [".git", ".gitignore"]
    ignorePatterns = set()
    ROOT = "."
    with open(os.path.join(ROOT, ".gitignore")) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith("#"):
                if line.endswith("\n"):
                    line = line[:-1]
                if line.endswith("/"):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print("Backup Finished!")


# FiLM-MP: small phase controller for anchor densification.
def mp_enabled(opt):
    return getattr(opt, "mp_growing", False)


def phase_at(iteration, opt):
    """Return the current densification phase."""
    if not mp_enabled(opt):
        return "base"
    if iteration <= opt.mp_coarse_until:
        return "coarse"
    if iteration <= opt.mp_fine_until:
        return "fine"
    if iteration <= opt.mp_prune_until:
        return "prune"
    return "refine"


def stat_on(iteration, opt):
    """Collect densification statistics only while grow/prune can use them."""
    if mp_enabled(opt):
        return opt.start_stat < iteration <= opt.mp_prune_until
    return opt.start_stat < iteration < opt.update_until


def grow_on(iteration, opt, phase):
    """Run anchor growing on the configured update interval."""
    return (
        phase in {"base", "coarse", "fine"}
        and iteration > opt.update_from
        and iteration % opt.update_interval == 0
    )


def prune_on(iteration, opt, phase):
    """Run pruning-only updates during the MP prune phase."""
    return phase == "prune" and iteration % opt.update_interval == 0


def free_now(iteration, opt):
    """Free densification buffers once anchors become fixed."""
    if mp_enabled(opt):
        return iteration > opt.mp_prune_until
    return iteration == opt.update_until


def training(
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    wandb=None,
    logger=None,
    ply_path=None,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.ratio,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        dataset.use_film_net,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if logger is not None:
        logger.info(
            "Ideas enabled: mp_growing={} use_film_net={}".format(
                getattr(opt, "mp_growing", False),
                getattr(dataset, "use_film_net", False),
            )
        )
        logger.info(
            "FiLM-MP: coarse<= {} fine<= {} prune<= {} refine>; opacity_confirm_ratio={}".format(
                getattr(opt, "mp_coarse_until", opt.update_until),
                getattr(opt, "mp_fine_until", opt.update_until),
                getattr(opt, "mp_prune_until", opt.update_until),
                getattr(opt, "mp_opa_confirm_ratio", 1.0),
            )
        )

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    training_time_start = time.time()
    last_phase = None
    stats_freed = False
    for iteration in range(first_iter, opt.iterations + 1):
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        phase = phase_at(iteration, opt)
        if mp_enabled(opt) and phase != last_phase:
            if phase in {"fine", "prune"}:
                gaussians.reset_densification_stats()
                if logger is not None:
                    logger.info(
                        "\n[ITER {}] FiLM-MP enters {} phase; reset densification stats".format(
                            iteration,
                            phase,
                        )
                    )
            last_phase = phase

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        stats_on = stat_on(iteration, opt)
        # FiLM-MP: coarse/base need 2D position gradients; fine also needs opacity gradients.
        retain_grad = stats_on and phase in {"base", "coarse", "fine"}
        retain_opacity_grad = stats_on and phase == "fine"
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
            retain_opacity_grad=retain_opacity_grad,
        )

        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            offset_selection_mask,
            radii,
            scaling,
            opacity,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["selection_mask"],
            render_pkg["radii"],
            render_pkg["scaling"],
            render_pkg["neural_opacity"],
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = 1.0 - ssim(image, gt_image)
        scaling_reg = scaling.prod(dim=1).mean()

        
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * ssim_loss
            + 0.01 * scaling_reg

        )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                dataset_name,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
                wandb,
                logger,
            )
            if iteration in saving_iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # FiLM-MP: collect stats for grow/prune phases only.
            if stats_on:
                # Coarse/base use position gradients; fine also stores opacity gradients.
                gaussians.training_statis(
                    viewspace_point_tensor,
                    opacity,
                    visibility_filter,
                    offset_selection_mask,
                    voxel_visible_mask,
                    use_grad=retain_grad,
                    use_opacity_grad=retain_opacity_grad,
                )

                # Fine phase confirms position candidates with opacity gradients.
                if grow_on(iteration, opt, phase):
                    gaussians.adjust_anchor(
                        check_interval=opt.update_interval,
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold,
                        min_opacity=opt.min_opacity,
                        grow_phase=phase,
                        opacity_confirm_ratio=opt.mp_opa_confirm_ratio,
                        allow_grow=True,
                        allow_prune=True,
                    )
                elif prune_on(iteration, opt, phase):
                    gaussians.prune_low_opacity(
                        check_interval=opt.update_interval,
                        success_threshold=opt.success_threshold,
                        min_opacity=opt.min_opacity,
                    )
            elif not stats_freed and free_now(iteration, opt):
                # FiLM-MP: refinement needs no densification buffers.
                gaussians.release_densification_stats()
                stats_freed = True
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            if iteration in checkpoint_iterations:
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    training_time_sec = time.time() - training_time_start
    if logger is not None:
        logger.info(
            "Training time: \033[1;35m{:.2f} sec ({:.2f} min)\033[0m".format(
                training_time_sec, training_time_sec / 60.0
            )
        )
    if tb_writer:
        tb_writer.add_scalar(f"{dataset_name}/training_time_sec", training_time_sec, opt.iterations)
    if wandb is not None:
        wandb.log({"training_time_sec": training_time_sec})


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def get_path_size_mb(path):
    if not path or not os.path.exists(path):
        return 0.0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024.0 * 1024.0)

    total_size = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
    return total_size / (1024.0 * 1024.0)


def get_model_size_mb(model_path, iteration):
    point_cloud_dir = os.path.join(
        model_path, "point_cloud", "iteration_{}".format(iteration)
    )
    return get_path_size_mb(point_cloud_dir)


def apply_render_options(gaussians, opt):
    return


def training_report(
    tb_writer,
    dataset_name,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    wandb=None,
    logger=None,
):
    if tb_writer:
        tb_writer.add_scalar(
            f"{dataset_name}/train_loss_patches/l1_loss", Ll1.item(), iteration
        )
        tb_writer.add_scalar(
            f"{dataset_name}/train_loss_patches/total_loss", loss.item(), iteration
        )
        tb_writer.add_scalar(f"{dataset_name}/iter_time", elapsed, iteration)

    if wandb is not None:
        wandb.log(
            {
                "train_l1_loss": Ll1,
                "train_total_loss": loss,
            }
        )

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config["cameras"]):
                    voxel_visible_mask = prefilter_voxel(
                        viewpoint, scene.gaussians, *renderArgs
                    )
                    image = torch.clamp(
                        renderFunc(
                            viewpoint,
                            scene.gaussians,
                            *renderArgs,
                            visible_mask=voxel_visible_mask,
                        )["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(
                            f"{dataset_name}/"
                            + config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{dataset_name}/"
                            + config["name"]
                            + "_view_{}/errormap".format(viewpoint.image_name),
                            (gt_image[None] - image[None]).abs(),
                            global_step=iteration,
                        )

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None] - image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{dataset_name}/"
                                + config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                logger.info(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )

                if tb_writer:
                    tb_writer.add_scalar(
                        f"{dataset_name}/"
                        + config["name"]
                        + "/loss_viewpoint - l1_loss",
                        l1_test,
                        iteration,
                    )
                    tb_writer.add_scalar(
                        f"{dataset_name}/" + config["name"] + "/loss_viewpoint - psnr",
                        psnr_test,
                        iteration,
                    )
                if wandb is not None:
                    wandb.log(
                        {
                            f"{config['name']}_loss_viewpoint_l1_loss": l1_test,
                            f"{config['name']}_PSNR": psnr_test,
                        }
                    )

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(
                f"{dataset_name}/" + "total_points",
                scene.gaussians.get_anchor.shape[0],
                iteration,
            )
        torch.cuda.empty_cache()

        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    neural_gaussian_count_list = []
    name_list = []
    per_view_dict = {}
    neural_gaussian_per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize()
        t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(
            view, gaussians, pipeline, background, visible_mask=voxel_visible_mask
        )
        torch.cuda.synchronize()
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        neural_gaussian_count = render_pkg["neural_gaussian_count"]
        visible_count_list.append(visible_count)
        neural_gaussian_count_list.append(neural_gaussian_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()

        name_list.append("{0:05d}".format(idx) + ".png")
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            errormap, os.path.join(error_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )
        image_name = "{0:05d}".format(idx) + ".png"
        per_view_dict[image_name] = visible_count.item()
        neural_gaussian_per_view_dict[image_name] = neural_gaussian_count

    with open(
        os.path.join(
            model_path, name, "ours_{}".format(iteration), "per_view_count.json"
        ),
        "w",
    ) as fp:
        json.dump(per_view_dict, fp, indent=True)

    with open(
        os.path.join(
            model_path, name, "ours_{}".format(iteration), "per_view_neural_gaussian_count.json"
        ),
        "w",
    ) as fp:
        json.dump(neural_gaussian_per_view_dict, fp, indent=True)

    return t_list, visible_count_list, neural_gaussian_count_list


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    opt=None,
    skip_train=True,
    skip_test=False,
    wandb=None,
    tb_writer=None,
    dataset_name=None,
    logger=None,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.use_film_net,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        apply_render_options(gaussians, opt)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        metrics = {
            "model_size_mb": get_model_size_mb(dataset.model_path, scene.loaded_iter),
            "visible_count": None,
            "neural_gaussian_count": None,
        }

        if not skip_train:
            t_train_list, train_visible_count, train_neural_gaussian_count = render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f"Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m")
            logger.info(
                "Train neural Gaussians/view: \033[1;35m{:.2f}\033[0m".format(
                    torch.tensor(train_neural_gaussian_count, dtype=torch.float32).mean().item()
                )
            )
            if wandb is not None:
                wandb.log(
                    {
                        "train_fps": train_fps.item(),
                        "train_neural_gaussians_per_view": torch.tensor(
                            train_neural_gaussian_count, dtype=torch.float32
                        ).mean().item(),
                    }
                )

        if not skip_test:
            t_test_list, test_visible_count, test_neural_gaussian_count = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f"Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m")
            logger.info(
                "Test neural Gaussians/view: \033[1;35m{:.2f}\033[0m".format(
                    torch.tensor(test_neural_gaussian_count, dtype=torch.float32).mean().item()
                )
            )
            logger.info(
                "Model size: \033[1;35m{:.3f} MB\033[0m".format(
                    metrics["model_size_mb"]
                )
            )
            metrics["visible_count"] = test_visible_count
            metrics["neural_gaussian_count"] = test_neural_gaussian_count
            if tb_writer:
                tb_writer.add_scalar(f"{dataset_name}/test_FPS", test_fps.item(), 0)
                tb_writer.add_scalar(
                    f"{dataset_name}/test_neural_gaussians_per_view",
                    torch.tensor(test_neural_gaussian_count, dtype=torch.float32).mean().item(),
                    0,
                )
                tb_writer.add_scalar(
                    f"{dataset_name}/model_size_mb",
                    metrics["model_size_mb"],
                    0,
                )
            if wandb is not None:
                wandb.log(
                    {
                        "test_fps": test_fps,
                        "test_neural_gaussians_per_view": torch.tensor(
                            test_neural_gaussian_count, dtype=torch.float32
                        ).mean().item(),
                        "model_size_mb": metrics["model_size_mb"],
                    }
                )

    return metrics


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(
    model_paths,
    render_metrics=None,
    wandb=None,
    tb_writer=None,
    dataset_name=None,
    logger=None,
):
    render_metrics = render_metrics or {}
    visible_count = render_metrics.get("visible_count")
    neural_gaussian_count = render_metrics.get("neural_gaussian_count")
    model_size_mb = render_metrics.get("model_size_mb", 0.0)

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log(
                {
                    "test_SSIMS": torch.stack(ssims).mean().item(),
                }
            )
            wandb.log(
                {
                    "test_PSNR_final": torch.stack(psnrs).mean().item(),
                }
            )
            wandb.log(
                {
                    "test_LPIPS": torch.stack(lpipss).mean().item(),
                }
            )

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info(
            "  SSIM : \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(ssims).mean(), ".5"
            )
        )
        logger.info(
            "  PSNR : \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(psnrs).mean(), ".5"
            )
        )
        logger.info(
            "  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(
                torch.tensor(lpipss).mean(), ".5"
            )
        )
        if neural_gaussian_count is not None:
            logger.info(
                "  Neural Gaussians/view: \033[1;35m{:>12.2f}\033[0m".format(
                    torch.tensor(neural_gaussian_count, dtype=torch.float32).mean().item()
                )
            )
        logger.info("  Model Size: \033[1;35m{:>12.3f} MB\033[0m".format(model_size_mb))
        print("")

        if tb_writer:
            tb_writer.add_scalar(
                f"{dataset_name}/SSIM", torch.tensor(ssims).mean().item(), 0
            )
            tb_writer.add_scalar(
                f"{dataset_name}/PSNR", torch.tensor(psnrs).mean().item(), 0
            )
            tb_writer.add_scalar(
                f"{dataset_name}/LPIPS", torch.tensor(lpipss).mean().item(), 0
            )

            if visible_count is not None:
                tb_writer.add_scalar(
                    f"{dataset_name}/VISIBLE_NUMS",
                    torch.tensor(visible_count).mean().item(),
                    0,
                )
            if neural_gaussian_count is not None:
                tb_writer.add_scalar(
                    f"{dataset_name}/NEURAL_GAUSSIANS",
                    torch.tensor(neural_gaussian_count, dtype=torch.float32).mean().item(),
                    0,
                )
            tb_writer.add_scalar(f"{dataset_name}/MODEL_SIZE_MB", model_size_mb, 0)

        full_dict[scene_dir][method].update(
            {
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
                "NEURAL_GAUSSIANS_PER_VIEW": None if neural_gaussian_count is None else torch.tensor(neural_gaussian_count, dtype=torch.float32).mean().item(),
                "MODEL_SIZE_MB": model_size_mb,
            }
        )
        per_view_metrics = {
            "SSIM": {
                name: ssim
                for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
            },
            "PSNR": {
                name: psnr
                for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)
            },
            "LPIPS": {
                name: lp
                for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)
            },
        }
        if visible_count is not None:
            per_view_metrics["VISIBLE_COUNT"] = {
                name: vc
                for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)
            }
        if neural_gaussian_count is not None:
            per_view_metrics["NEURAL_GAUSSIAN_COUNT"] = {
                name: int(count)
                for count, name in zip(neural_gaussian_count, image_names)
            }
        per_view_dict[scene_dir][method].update(per_view_metrics)

    with open(scene_dir + "/results.json", "w") as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", "w") as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--warmup", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="-1")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f"args: {args}")

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f"using GPU {args.gpu}")

    try:
        saveRuntimeCode(os.path.join(args.model_path, "backup"))
    except:
        logger.info(f"save code failed~")

    dataset = args.source_path.split("/")[-1]
    exp_name = args.model_path.split("/")[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args),
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        dataset,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        wandb,
        logger,
    )
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(
            args.model_path,
            f"point_cloud/iteration_{args.iterations}",
            "point_cloud.ply",
        )
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            dataset,
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            wandb=wandb,
            logger=logger,
            ply_path=new_ply_path,
        )

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f"\nStarting Rendering~")
    render_metrics = render_sets(
        lp.extract(args), -1, pp.extract(args), opt=op.extract(args), wandb=wandb, logger=logger
    )
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, render_metrics=render_metrics, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
