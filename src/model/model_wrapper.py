from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.encoder_freesplat import UseDepthMode
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import mmcv
import os
import json


def convert_array_to_pil(depth_map, no_text=False):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    # Ensure depth_map is 2D
    if len(depth_map.shape) > 2:
        depth_map = depth_map.squeeze()

    mask = depth_map!=0
    disp_map = 1/(depth_map + 1e-8)

    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask_ = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask_] = 255
    min_depth, max_depth = depth_map[mask].min(), depth_map[mask].max()
    image = Image.fromarray(colormapped_im)
    if not no_text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)
        draw.text((20,20), '[%.2f, %.2f]'%(min_depth, max_depth), (255,255,255), font=font)
        colormapped_im = np.asarray(image)

    return colormapped_im

# Compute RGB metrics at novel views.
def compute_metrics(rgb_gt, rgb):
    rgb = rgb.clip(min=0, max=1)
    psnr = compute_psnr(rgb_gt, rgb)
    if rgb_gt.shape[0] < 100:
        lpips = compute_lpips(rgb_gt, rgb)
    else:
        lpips = torch.tensor(0.0, device=rgb.device)
    ssim = compute_ssim(rgb_gt, rgb)
    num = len(psnr)
    psnr = psnr.mean().item()
    ssim = ssim.mean().item()
    lpips = lpips.mean().item()
    print('psnr:', psnr, 'ssim:', ssim, 'lpips:', lpips, 'num:', num)
    return psnr, lpips, ssim, num

# Compute Depth metrics at novel views.
def depth_render_metrics(prediction, batch) -> Float[Tensor, ""]:
    if not 'depth' in batch['target']:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    target = batch['target']['depth'].squeeze(2)
    gt_bN = rearrange(target.clone(), 'b v h w -> (b v) (h w)')
    pred_bN = rearrange(prediction.depth.clone(), 'b v h w -> (b v) (h w)')
    mask = gt_bN > 0.5
    gt_bN[~mask] = torch.nan
    pred_bN[~mask] = torch.nan
    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1).mean()
    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1).mean()
    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a25_val = (thresh_bN < (1.0+0.25)     ).float()
    a25_val[~mask] = torch.nan
    delta_25 = torch.nanmean(a25_val, dim=1).mean()

    a10_val = (thresh_bN < (1.0+0.1)     ).float()
    a10_val[~mask] = torch.nan
    delta_10 = torch.nanmean(a10_val, dim=1).mean()
    return abs_diff_b, abs_rel_b, delta_25, delta_10


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    save_image: bool = True
    save_input_images: bool = True
    save_gt_image: bool = True
    save_video: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    load_depth: UseDepthMode | None
    extended_visualization: bool
    has_depth: bool = False


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        cfg_dict,
        run_dir,
        num_context_views: int = 2,
        dataset_name: str = 'scannet',
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.run_dir = run_dir
        self.num_context_views = num_context_views
        self.dataset_name = dataset_name

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.losses_log = {}
        self.loss_total = []
        self.metrics = {}
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric] = []
        self.num_evals = []
        self.test_scene_list = []
        self.test_fvs_list = []

        for k1 in cfg_dict:
            try:
                keys = cfg_dict[k1].keys()
                print(f'{k1}:')
                for k2 in keys:
                    print(f'    {k2}: {cfg_dict[k1][k2]}')
            except:
                print(f'{k1}: {cfg_dict[k1]}')

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        B, _, _, h, w = batch["target"]["image"].shape
        
        encoder_results = self.encoder(batch["context"], self.global_step, False, is_testing=False)
        
        gaussians = encoder_results['gaussians']

        if not isinstance(gaussians, list):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
        else:
            output_list = []
            for i, gs in enumerate(gaussians):
                output_list.append(self.decoder.forward(
                    gs,
                    batch["target"]["extrinsics"][i:i+1],
                    batch["target"]["intrinsics"][i:i+1],
                    batch["target"]["near"][i:i+1],
                    batch["target"]["far"][i:i+1],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                ))
            output = DecoderOutput(None, None)
            output.color = torch.cat([x.color for x in output_list], dim=0)
            try:
                output.depth = torch.cat([x.depth for x in output_list], dim=0)
            except:
                pass
        output_dr = None
        target_gt = batch["target"]["image"]

        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        psnr = psnr_probabilistic.mean()
        self.log("train/psnr", psnr, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, encoder_results, self.global_step, output_dr)
            self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
            total_loss = total_loss + loss
            self.losses_log[loss_fn.name] = self.losses_log.get(loss_fn.name, [])
            self.losses_log[loss_fn.name].append(loss)
        self.log("loss/total", total_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.loss_total.append(total_loss)
        context_indices = batch['context']['index'].tolist()

        if batch_idx %10 == 0:
            to_print = f"train step {self.global_step}; "+\
                       f"scene = {batch['scene']}; " + \
                       f"context = {context_indices}; " +\
                       f"loss = {torch.mean(torch.tensor(self.loss_total)):.6f} "+\
                       f"psnr = {torch.mean(torch.tensor(psnr)):.2f}"
            for name in self.losses_log:
                to_print = to_print + f' loss_{name} = {torch.mean(torch.tensor(self.losses_log[name])):.6f}'
            if 'gs_ratio' in encoder_results:
                to_print = to_print + f' gs_ratio = {torch.mean(torch.tensor(encoder_results["gs_ratio"])):.6f}'
            print(to_print)
            self.losses_log = {}
            self.loss_total = []
            
            
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def on_test_start(self):
        self.test_step_count = 0
        self.early_stopped = False
        # 初始化指标收集器
        self.all_metrics = {
            'psnr_inter': [],
            'lpips_inter': [],
            'ssim_inter': [],
            'num_inter': [],
            'depth_abs_diff': [],
            'depth_rel_diff': [],
            'depth_delta_25': [],
            'depth_delta_10': [],
            'num_gaussians': [],
            # 添加其他需要的指标
        }

    def test_step(self, batch, batch_idx):
        # 步数计数器递增
        print(f"Test step count before increment: {self.test_step_count}")
        self.test_step_count += 1

        batch: BatchedExample = self.data_shim(batch)
        # 检查是否达到650步且尚未提前结束
        if self.test_step_count >= 650 and not self.early_stopped:
            print(f"Reached 650 steps at batch {batch_idx}, triggering early test end")
            self.early_stopped = True
            
            # 直接执行测试结束逻辑
            self.execute_test_end()
            
            # 设置训练器停止标志
            self.trainer.should_stop = True
            
            # 跳过当前步骤的剩余处理
            return
        

        b, v, _, h, w = batch["target"]["image"].shape
        print("target_views:", v)
        _, v_c, _, _, _= batch["context"]["image"].shape
        print("context_views:", v_c)

        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
    
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            encoder_results = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                is_testing=True,
                export_ply=self.encoder_visualizer.cfg.export_ply,
            )
            gaussians = encoder_results['gaussians']
            
        with self.benchmarker.time("decoder", num_calls=v):
            if not isinstance(gaussians, list):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode='depth',
                )
            else:
                output_list = []
                n_targets = batch["target"]["extrinsics"].shape[1]
                for i, gs in enumerate(gaussians):
                    output = []
                    for j in range(np.ceil(n_targets/50).astype(int)):
                        output.append(self.decoder.forward(
                            gs,
                            batch["target"]["extrinsics"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["intrinsics"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["near"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["far"][i:i+1, j*50:(j+1)*50],
                            (h, w),
                            depth_mode='depth',
                        ))
                    now = DecoderOutput(None, None)
                    now.color = torch.cat([x.color for x in output], dim=1)
                    now.depth = torch.cat([x.depth for x in output], dim=1)
                    output_list.append(now)
                output = DecoderOutput(None, None)
                output.color = torch.cat([x.color for x in output_list], dim=0)
                try:
                    output.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass

        # Save images.
        (scene,) = batch["scene"]
        print(f'processing {scene}')
        self.test_scene_list.append(scene)
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        # Enhanced image saving based on BEV-Splat implementation
        from pathlib import Path
        import os
        base_path = Path(get_cfg()["output_dir"])

        # Check if image saving is enabled
        save_images_enabled = self.test_cfg.save_image
        save_input_images_enabled = self.test_cfg.save_input_images
        save_gt_images_enabled = self.test_cfg.save_gt_image
        save_video_enabled = self.test_cfg.save_video

        # Save input context images for visualization
        if save_input_images_enabled:
            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_dir = base_path / "images" / scene / "color"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_image(color, save_dir / f"input_{idx:0>6}.png")
        abs_diff, rel_diff, delta_25, delta_10 = depth_render_metrics(output, batch)
        print(f'abs_diff: {abs_diff}, rel_diff: {rel_diff}, delta_25: {delta_25}, delta_10: {delta_10}')
        self.benchmarker.store('depth_abs_diff', float(abs_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_rel_diff', float(rel_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_delta_25', float(delta_25.detach().cpu().numpy()))
        self.benchmarker.store('depth_delta_10', float(delta_10.detach().cpu().numpy()))

        try:
            fvs_length = batch["target"]["test_fvs"]
            test_fvs = fvs_length > 0
        except:
            fvs_length = 0
            test_fvs = False
            
        count = 0
        for i, index, fig in zip(range(len(batch["context"]["index"][0])), batch["context"]["index"][0], batch["context"]["image"][0]):
            length = len(encoder_results[f"depth_num0_s-1"][0])
            save_image(fig, path / scene / f"contexts/{index:0>6}.png")
            save_image(torch.from_numpy(convert_array_to_pil(encoder_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h,w), no_text=True).transpose(2,0,1)\
                                        .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                        path / scene / f"depth_pred/{index:0>6}.png")
            

        
        for i, index in enumerate(batch["target"]["index"][0]):
            depth_render = output.depth[0][i]

            save_image(torch.from_numpy(convert_array_to_pil(depth_render.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                            .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                            path / scene / f"depth_render/{index:0>6}.png")
            
            if 'depth' in batch["target"]:
                depth_gt = batch["target"]['depth'][0][i]
                
                save_image(torch.from_numpy(convert_array_to_pil(depth_gt.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                                .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                                path / scene / f"depth_render_gt/{index:0>6}.png")

        # Enhanced rendering image saving
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save rendered and ground truth images with better organization
        if save_images_enabled:
            for index, color, color_gt in zip(batch["target"]["index"][0], images_prob, rgb_gt):
                if not test_fvs:
                    # Save rendered images
                    color_dir = base_path / "images" / scene / "color"
                    color_dir.mkdir(parents=True, exist_ok=True)
                    save_image(color, color_dir / f"{index:0>6}.png")

                    # Save ground truth images if enabled
                    if save_gt_images_enabled:
                        save_image(color_gt, color_dir / f"{index:0>6}_gt.png")

                    # Also save to original path for compatibility
                    save_image(color, path / scene / f"color/{index:0>6}.png")
                    save_image(color_gt, path / scene / f"color_gt/{index:0>6}.png")
                else:
                    if count < batch["target"]["index"][0].shape[0]-fvs_length:
                        interp_dir = base_path / "images" / scene / "interpolation"
                        interp_dir.mkdir(parents=True, exist_ok=True)
                        save_image(color, interp_dir / f"{index:0>6}.png")
                        if save_gt_images_enabled:
                            save_image(color_gt, interp_dir / f"{index:0>6}_gt.png")

                        # Original paths for compatibility
                        save_image(color, path / scene / f"interpolation/{index:0>6}.png")
                        save_image(color_gt, path / scene / f"interapolation_gt/{index:0>6}.png")
                    else:
                        extrap_dir = base_path / "images" / scene / "extrapolation"
                        extrap_dir.mkdir(parents=True, exist_ok=True)
                        save_image(color, extrap_dir / f"{index:0>6}.png")
                        if save_gt_images_enabled:
                            save_image(color_gt, extrap_dir / f"{index:0>6}_gt.png")

                        # Original paths for compatibility
                        save_image(color, path / scene / f"extrapolation/{index:0>6}.png")
                        save_image(color_gt, path / scene / f"extrapolation_gt/{index:0>6}.png")
                    count += 1

        # Save video if enabled
        if save_video_enabled:
            try:
                from ..misc.image_io import save_video
                frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
                video_dir = base_path / "videos"
                video_dir.mkdir(parents=True, exist_ok=True)
                save_video(
                    [img for img in images_prob],
                    video_dir / f"{scene}_frame_{frame_str}.mp4",
                )
                print(f"Video saved: {video_dir / f'{scene}_frame_{frame_str}.mp4'}")
            except Exception as e:
                print(f"Warning: Failed to save video: {e}")
        
        if not test_fvs:
            psnr, lpips, ssim, num = compute_metrics(batch["target"]["image"][0], output.color[0])
        
            self.benchmarker.store('psnr_inter', float(psnr))
            self.benchmarker.store('lpips_inter', float(lpips))
            self.benchmarker.store('ssim_inter', float(ssim))
            self.benchmarker.store('num_inter', float(num))
            self.benchmarker.store('num_gaussians', encoder_results['num_gaussians'])
            self.test_fvs_list.append(False)
        else:
            length = batch["target"]["index"][0].shape[0]
            psnr_inter, lpips_inter, ssim_inter, num_inter = compute_metrics(batch["target"]["image"][0][:length-fvs_length], 
                                                                  output.color[0][:length-fvs_length])
            psnr_extra, lpips_extra, ssim_extra, num_extra = compute_metrics(batch["target"]["image"][0][length-fvs_length:],
                                                                  output.color[0][length-fvs_length:])
            
            self.benchmarker.store('psnr_inter', float(psnr_inter))
            self.benchmarker.store('lpips_inter', float(lpips_inter))
            self.benchmarker.store('ssim_inter', float(ssim_inter))
            self.benchmarker.store('num_inter', float(num_inter))
            self.benchmarker.store('psnr_extra', float(psnr_extra))
            self.benchmarker.store('lpips_extra', float(lpips_extra))
            self.benchmarker.store('ssim_extra', float(ssim_extra))
            self.benchmarker.store('num_extra', float(num_extra))
            self.benchmarker.store('num_gaussians', encoder_results['num_gaussians'])
            self.test_fvs_list.append(True)
        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                encoder_results, batch["context"], batch_idx, out_path=self.test_cfg.output_path / 'gaussians'
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

         # 收集指标
        self.all_metrics['psnr_inter'].append(float(psnr))
        self.all_metrics['lpips_inter'].append(float(lpips))
        self.all_metrics['ssim_inter'].append(float(ssim))
        self.all_metrics['num_inter'].append(float(num))
        self.all_metrics['depth_abs_diff'].append(float(abs_diff.detach().cpu().numpy()))
        self.all_metrics['depth_rel_diff'].append(float(rel_diff.detach().cpu().numpy()))
        self.all_metrics['depth_delta_25'].append(float(delta_25.detach().cpu().numpy()))
        self.all_metrics['depth_delta_10'].append(float(delta_10.detach().cpu().numpy()))
        self.all_metrics['num_gaussians'].append(encoder_results['num_gaussians'])

        # Save scene-specific metrics to file (similar to BEV-Splat implementation)
        if save_images_enabled:
            try:
                import json
                metrics_file = base_path / "scene_metrics.txt"
                with open(metrics_file, "a") as f:
                    f.write(f"{scene}: psnr={psnr:.4f}, lpips={lpips:.4f}, ssim={ssim:.4f}\n")

                # 将scene和lpips写入txt文件
                with open(base_path / "scene_lpips.txt", "a") as f:
                    f.write(f"{scene}: {lpips:.6f}\n")

                # Also save detailed metrics as JSON
                scene_metrics = {
                    "scene": scene,
                    "psnr_inter": float(psnr),
                    "lpips_inter": float(lpips),
                    "ssim_inter": float(ssim),
                    "num_inter": float(num),
                    "depth_abs_diff": float(abs_diff.detach().cpu().numpy()),
                    "depth_rel_diff": float(rel_diff.detach().cpu().numpy()),
                    "depth_delta_25": float(delta_25.detach().cpu().numpy()),
                    "depth_delta_10": float(delta_10.detach().cpu().numpy()),
                    "num_gaussians": encoder_results['num_gaussians']
                }

                scene_json_file = base_path / "scene_detailed_metrics.jsonl"
                with open(scene_json_file, "a") as f:
                    f.write(json.dumps(scene_metrics) + "\n")

            except Exception as e:
                print(f"Warning: Failed to save scene metrics: {e}")


    def execute_test_end(self):
        """计算并打印所有收集到的指标的平均值"""
        print("\n" + "="*50)
        print("Early Termination at Step 650 - Final Metrics")
        print("="*50)
        
        # 确保所有指标都有数据
        for key in self.all_metrics:
            if len(self.all_metrics[key]) == 0:
                print(f"Warning: No data for metric {key}")
                return
        
        # 计算简单平均值（非加权）
        psnr_inter_avg = sum(self.all_metrics['psnr_inter']) / len(self.all_metrics['psnr_inter'])
        lpips_inter_avg = sum(self.all_metrics['lpips_inter']) / len(self.all_metrics['lpips_inter'])
        ssim_inter_avg = sum(self.all_metrics['ssim_inter']) / len(self.all_metrics['ssim_inter'])
        
        # 计算深度指标平均值
        depth_abs_diff_avg = sum(self.all_metrics['depth_abs_diff']) / len(self.all_metrics['depth_abs_diff'])
        depth_rel_diff_avg = sum(self.all_metrics['depth_rel_diff']) / len(self.all_metrics['depth_rel_diff'])
        depth_delta_25_avg = sum(self.all_metrics['depth_delta_25']) / len(self.all_metrics['depth_delta_25'])
        depth_delta_10_avg = sum(self.all_metrics['depth_delta_10']) / len(self.all_metrics['depth_delta_10'])
        
        # 计算高斯数量平均值
        num_gaussians_avg = sum(self.all_metrics['num_gaussians']) / len(self.all_metrics['num_gaussians'])
        
        # 打印结果
        print(f"PSNR (Interpolation): {psnr_inter_avg:.4f}")
        print(f"LPIPS (Interpolation): {lpips_inter_avg:.4f}")
        print(f"SSIM (Interpolation): {ssim_inter_avg:.4f}")
        print(f"Depth Abs Diff: {depth_abs_diff_avg:.4f}")
        print(f"Depth Rel Diff: {depth_rel_diff_avg:.4f}")
        print(f"Depth Delta 25: {depth_delta_25_avg:.4f}")
        print(f"Depth Delta 10: {depth_delta_10_avg:.4f}")
        print(f"Avg Gaussians: {num_gaussians_avg:.0f}")
        
        # 如果有外推指标
        if 'psnr_extra' in self.all_metrics and len(self.all_metrics['psnr_extra']) > 0:
            psnr_extra_avg = sum(self.all_metrics['psnr_extra']) / len(self.all_metrics['psnr_extra'])
            lpips_extra_avg = sum(self.all_metrics['lpips_extra']) / len(self.all_metrics['lpips_extra'])
            ssim_extra_avg = sum(self.all_metrics['ssim_extra']) / len(self.all_metrics['ssim_extra'])
                
            print(f"\nExtrapolation Metrics:")
            print(f"PSNR (Extrapolation): {psnr_extra_avg:.4f}")
            print(f"LPIPS (Extrapolation): {lpips_extra_avg:.4f}")
            print(f"SSIM (Extrapolation): {ssim_extra_avg:.4f}")
        
        print("="*50 + "\n")
        
        # 确保测试结束后也调用原始的on_test_end
        self.on_test_end()


    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.dump_stats(
            self.test_cfg.output_path / name / "stats.json"
        )
        for i in range(min(len(self.test_scene_list), len(self.test_fvs_list))):
            if self.test_fvs_list[i]:
                print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                self.benchmarker.benchmarks['ssim_inter'][i],
                                self.benchmarker.benchmarks['lpips_inter'][i],
                                self.benchmarker.benchmarks['psnr_extra'][int(np.sum(self.test_fvs_list[:i]))], 
                                self.benchmarker.benchmarks['ssim_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                self.benchmarker.benchmarks['lpips_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                self.benchmarker.benchmarks['depth_abs_diff'][i],
                                self.benchmarker.benchmarks['depth_rel_diff'][i],
                                self.benchmarker.benchmarks['depth_delta_25'][i],
                                self.benchmarker.benchmarks['depth_delta_10'][i])
            else:
                print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                self.benchmarker.benchmarks['ssim_inter'][i],
                                self.benchmarker.benchmarks['lpips_inter'][i],
                                self.benchmarker.benchmarks['depth_abs_diff'][i],
                                self.benchmarker.benchmarks['depth_rel_diff'][i],
                                self.benchmarker.benchmarks['depth_delta_25'][i],
                                self.benchmarker.benchmarks['depth_delta_10'][i])
        print('psnr_inter_avg:', (np.array(self.benchmarker.benchmarks['psnr_inter']) 
                                * np.array(self.benchmarker.benchmarks['num_inter'])).sum()
                                / np.array(self.benchmarker.benchmarks['num_inter']).sum(), 
            'ssim_inter_avg:', (np.array(self.benchmarker.benchmarks['ssim_inter']) 
                                * np.array(self.benchmarker.benchmarks['num_inter'])).sum()
                                / np.array(self.benchmarker.benchmarks['num_inter']).sum(),
            'lpips_inter_avg:', (np.array(self.benchmarker.benchmarks['lpips_inter']) 
                                * np.array(self.benchmarker.benchmarks['num_inter'])).sum()
                                / np.array(self.benchmarker.benchmarks['num_inter']).sum(),
            'depth_abs_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_abs_diff'])).cpu().numpy(),
            'depth_rel_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_rel_diff'])).cpu().numpy(),
            'depth_delta_25_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_delta_25'])).cpu().numpy(),
            'depth_delta_10_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_delta_10'])).cpu().numpy())
        try:
            print('psnr_extra_avg:', (np.array(self.benchmarker.benchmarks['psnr_extra']) 
                                * np.array(self.benchmarker.benchmarks['num_extra'])).sum()
                                / np.array(self.benchmarker.benchmarks['num_extra']).sum(), 
                'ssim_extra_avg:', (np.array(self.benchmarker.benchmarks['ssim_extra']) 
                                    * np.array(self.benchmarker.benchmarks['num_extra'])).sum()
                                    / np.array(self.benchmarker.benchmarks['num_extra']).sum(),
                'lpips_extra_avg:', (np.array(self.benchmarker.benchmarks['lpips_extra']) 
                                    * np.array(self.benchmarker.benchmarks['num_extra'])).sum()
                                    / np.array(self.benchmarker.benchmarks['num_extra']).sum())
        except:
            pass
        print('num_gaussians_avg:', self.benchmarker.benchmarks['num_gaussians_avg'])

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        context_indices = batch['context']['index'].tolist()

        (scene,) = batch["scene"]
        print(
            f"validation step {self.global_step}; "
            f"scene = {batch['scene']}; "
            f"context = {context_indices}"
        )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        encoder_probabilistic_results = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
            is_testing=True,
        )
        gaussians_probabilistic = encoder_probabilistic_results['gaussians']
        if not isinstance(gaussians_probabilistic, list):
            output_probabilistic = self.decoder.forward(
                gaussians_probabilistic,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode='depth',
            )
        else:
            output_probabilistic_list = []
            for i, gs in enumerate(gaussians_probabilistic):
                output_probabilistic_list.append(self.decoder.forward(
                    gs,
                    batch["target"]["extrinsics"][i:i+1],
                    batch["target"]["intrinsics"][i:i+1],
                    batch["target"]["near"][i:i+1],
                    batch["target"]["far"][i:i+1],
                    (h, w),
                    depth_mode='depth',
                ))
            output_probabilistic = DecoderOutput(None, None)
            output_probabilistic.color = torch.cat([x.color for x in output_probabilistic_list], dim=0)
            try:
                output_probabilistic.depth = torch.cat([x.depth for x in output_probabilistic_list], dim=0)
            except:
                pass
        output_dr = None
        rgb_probabilistic = output_probabilistic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        tag, rgb = "probabilistic", rgb_probabilistic
        psnr, lpips, ssim, num = compute_metrics(rgb_gt, rgb)
        self.log(f"val/psnr_{tag}", psnr)
        self.log(f"val/lpips_{tag}", lpips)
        self.log(f"val/ssim_{tag}", ssim)
        abs_diff, rel_diff, delta_25, delta_10 = depth_render_metrics(output_probabilistic, batch)
        self.log(f"val/depth_abs_diff_{tag}", abs_diff)
        self.log(f"val/depth_rel_diff_{tag}", rel_diff)
        self.log(f"val/depth_delta_25_{tag}", delta_25)
        self.log(f"val/depth_delta_10_{tag}", delta_10)
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric].append(eval(metric))
        self.num_evals.append(num)

        # Construct comparison image.
        if not self.train_cfg.has_depth:
            context_figs = []
            for fig in batch["context"]["image"][0]:
                context_figs.append(fig)
            if 'depth' in batch["context"]:
                for fig in batch["context"]["depth"][0]:
                    context_figs.append(torch.from_numpy(convert_array_to_pil(fig.cpu().numpy()[0]).transpose(2,0,1)\
                                                        .astype(np.float32)/255).to(batch["context"]["image"][0].device))
            comparison = hcat(
                add_label(vcat(*context_figs), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            )
        else:
            context_figs = []
            
            context_depth_figs = []
            pred_depth_figs = []
            for fig in batch["context"]["image"][0]:
                context_figs.append(fig)
                length = len(encoder_probabilistic_results[f"depth_num0_s-1"][0])
                for i in range(length):
                    try:
                        context_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(batch["context"][f"depth_s-1"][0][i][0].cpu().numpy(), (w,h),interpolation='nearest')).transpose(2,0,1)\
                                                                .astype(np.float32)/255).to(batch["context"]["image"][0].device))
                    except:
                        pass
                    try:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(encoder_probabilistic_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h,w)).transpose(2,0,1)\
                                                            .astype(np.float32)/255).to(batch["context"]["image"][0].device))
                    except:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(encoder_probabilistic_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h//(2**(s+1)), w//(2**(s+1))), (w,h),interpolation='nearest'))\
                                                                        .transpose(2,0,1).astype(np.float32)/255).to(batch["context"]["image"][0].device))

            try:
                comparison = hcat(
                add_label(vcat(*context_figs), "Context"),
                add_label(vcat(*context_depth_figs), "Context GT Depths"),
                add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
            )
            except:
                comparison = hcat(
                    add_label(vcat(*context_figs), "Context"),
                    add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
                )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
  

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                encoder_probabilistic_results, batch["context"], self.global_step, out_path=self.test_cfg.output_path
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)


    def on_validation_end(self) -> None:
        with open(self.run_dir + "/val_metrics.txt", "a") as f:
            line = '' 
            for metric in ['psnr', 'lpips', 'ssim']:
                try:
                    line = line + f'{metric}=' + str((np.array(self.metrics[metric])*np.array(self.num_evals)).sum() / np.array(self.num_evals).sum()) + ' '
                except:
                    pass
            f.write(line + '\n')
            print(line)
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric] = []
            self.num_evals = []

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False, is_testing=False)['gaussians']
        gaussians_det = self.encoder(batch["context"], self.global_step, True, is_testing=False)['gaussians']

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 1,
                            pct_start=0.001,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
