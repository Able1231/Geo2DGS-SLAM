""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from gc import is_tracked
from pathlib import Path
import torchvision
import numpy as np
import torch
import cv2
import torch.nn.functional as F

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.tracker_utils import extrapolate_poses
from src.utils.mapper_utils import exceeds_motion_thresholds
from src.utils.utils import get_render_settings, np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging
from src.entities.BA import BA
from src.utils.eval_pose import full_traj_eval
from tqdm import tqdm
from src.modules.droid_net import DroidNet
from src.entities.depth_video import DepthVideo
from src.entities.backend import Backend
from src.entities.motion_filter import MotionFilter
from src.entities.frontend import DroidFrontend
from src.entities.trajectory_filler import PoseTrajectoryFiller
from src.utils.pose_transform import transform_to_quaternion, quaternion_to_transform_noBatch, matrix_to_quaternion
class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.droid_net: DroidNet = DroidNet()
        self.load_pretrained(config)
        self.droid_net.to(self.device).eval()

        self.video = DepthVideo(self.config, buffer = self.config["tracking"]["buffer"])
        self.only_tracking = config.get("only_tracking", False)
        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        self.dataset = get_dataset(config["dataset_name"])({**config["data"], **config["cam"]})

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]

        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)

        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)

        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.ba = BA(config, self.dataset, self.logger)

        self.transform = torchvision.transforms.ToTensor()

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.config['tracking']['frontend']['window']
        filter_thresh = self.config['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.droid_net, self.video, filter_thresh)
        self.enable_online_ba = self.config['tracking']['frontend']['enable_online_ba']
        # frontend process
        self.frontend = DroidFrontend(self.droid_net, self.video, self.config)
        self.online_ba = Backend(self.droid_net, self.video, self.config)
        self.ba_freq = self.config['tracking']['backend']['ba_freq']

        self.new_submap_every_kf = config["mapping"].get("new_submap_every_kf", 10)
        self.last_submap_kf_idx = config["tracking"]["warmup"]

        self.saved_submaps = []

        self.intrinsic = torch.as_tensor(
            [self.config['cam']['fx'], self.config['cam']['fy'], self.config['cam']['cx'], self.config['cam']['cy']])

        self.W_edge = self.config['cam']['W_edge']
        self.H_edge = self.config['cam']['H_edge']

        self.W_out_with_edge = self.config['cam']['W_out'] + self.W_edge * 2
        self.H_out_with_edge = self.config['cam']['H_out'] + self.H_edge * 2

        self.intrinsic[0] *= self.W_out_with_edge / self.config['cam']['W']
        self.intrinsic[1] *= self.H_out_with_edge / self.config['cam']['H']
        self.intrinsic[2] *= self.W_out_with_edge / self.config['cam']['W']
        self.intrinsic[3] *= self.H_out_with_edge / self.config['cam']['H']


        self.intrinsic[2] -= self.W_edge
        self.intrinsic[3] -= self.H_edge

        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])

        self.traj_filler = PoseTrajectoryFiller(self.config, net=self.droid_net, video=self.video)

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        print(f"Load droid pretrained checkpoint from {droid_pretrained}!")

    @torch.no_grad()
    def _sync_mapper_poses(self, gaussian_model: GaussianModel) -> None:
        """ Updates the poses (``w2c`` and ``render_settings``) stored in the mapper's
        keyframes to follow the latest DROID-optimized poses in the depth video, and
        applies the pose delta to the gaussians anchored to each keyframe. The mapper's
        historical poses are otherwise fixed and become stale whenever the frontend or
        the global BA updates ``video.poses``.

        Keyframes whose pose has been dropped by the factor graph are removed from the
        mapper, and their gaussians are pruned so no gaussian is left anchored to a dead pose.
        """
        crop = self.mapper.mapping_crop
        intrinsics = self.mapper.get_mapping_intrinsics()
        height = self.dataset.height - 2 * crop
        width = self.dataset.width - 2 * crop

        # Partition keyframes into those still present in the depth video and those whose
        # pose was dropped by the factor graph; the latter get pruned.
        kept_keyframes = []
        gone_frame_ids = []
        for frame_id, keyframe in self.mapper.keyframes:
            if self._frame_id_to_kf_idx(frame_id) is None:
                gone_frame_ids.append(frame_id)
            else:
                kept_keyframes.append((frame_id, keyframe))

        if gone_frame_ids:
            self._prune_gaussians_by_frame_ids(gaussian_model, gone_frame_ids)
            for frame_id in gone_frame_ids:
                self.keyframes_info.pop(frame_id, None)
            self.mapper.keyframes = kept_keyframes

        for frame_id, keyframe in kept_keyframes:
            new_c2w = self._get_c2w_by_frame_id(frame_id).float().cuda()  # [4, 4]
            old_w2c = keyframe["w2c"].detach().float()                    # [4, 4] cuda
            old_c2w = torch.linalg.inv(old_w2c)                           # [4, 4] cuda

            # World-space delta of this camera: new_c2w @ inv(old_c2w)
            delta = new_c2w @ torch.linalg.inv(old_c2w)

            # Move/rotate the gaussians anchored to this keyframe by the delta.
            self._apply_pose_delta_to_gaussians(gaussian_model, frame_id, delta)

            # Update the stored keyframe pose.
            w2c = torch.linalg.inv(new_c2w).cpu().numpy()
            keyframe["w2c"] = torch.from_numpy(w2c).to('cuda')
            keyframe["render_settings"] = get_render_settings(width, height, intrinsics, w2c)

    @torch.no_grad()
    def _apply_delta_to_xyz_rot(self, xyz: torch.Tensor, rotation: torch.Tensor,
                                camera_id: torch.Tensor, frame_id: int,
                                delta: torch.Tensor) -> None:
        """ In-place world-space SE3 delta (new_c2w @ inv(old_c2w)) applied to the position
        and rotation of every gaussian whose ``camera_id`` matches ``frame_id``.
        ``xyz``/``rotation`` are mutated in place (either the active model's ``.data`` tensors
        or the detached CPU tensors of a saved submap).
        """
        mask = camera_id == frame_id
        if not mask.any():
            return

        R = delta[:3, :3]  # [3, 3]
        t = delta[:3, 3]   # [3]

        # position: xyz_new = R @ xyz + t
        xyz[mask] = xyz[mask] @ R.t() + t

        # rotation: q_new = q_delta (x) q_old  (Hamilton product, wxyz order)
        q_delta = matrix_to_quaternion(R)                                # [4] (w, x, y, z)
        q_old = torch.nn.functional.normalize(rotation[mask], dim=-1)    # [M, 4]
        w0, x0, y0, z0 = q_delta
        w1, x1, y1, z1 = q_old[:, 0], q_old[:, 1], q_old[:, 2], q_old[:, 3]
        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        q_new = torch.stack([w, x, y, z], dim=-1)
        rotation[mask] = torch.nn.functional.normalize(q_new, dim=-1)

    @torch.no_grad()
    def _apply_pose_delta_to_gaussians(self, gaussian_model: GaussianModel,
                                       frame_id: int, delta: torch.Tensor) -> None:
        """ Applies a world-space SE3 delta to the active model's gaussians anchored to
        ``frame_id``.
        """
        self._apply_delta_to_xyz_rot(
            gaussian_model._xyz.data, gaussian_model._rotation.data,
            gaussian_model._camera_id, frame_id, delta)

    @staticmethod
    def _is_tracking_only(gaussian_model: GaussianModel) -> bool:
        """ Returns True when the gaussian model's optimizer is in camera-tracking mode (only
        the camera pose is optimized), i.e. the last setup was ``training_setup_camera`` rather
        than the mapping ``training_setup``.
        """
        optimizer = getattr(gaussian_model, "optimizer", None)
        if optimizer is None:
            return False
        names = {pg.get("name") for pg in optimizer.param_groups if pg.get("name") is not None}
        return "cam_unnorm_rot" in names or "cam_trans" in names

    @torch.no_grad()
    def _prune_gaussians_by_frame_ids(self, gaussian_model: GaussianModel, frame_ids) -> int:
        """ Removes the gaussians anchored to ``frame_ids`` (whose poses were dropped from the
        depth video). Chooses the deletion method based on the current optimizer mode: the
        optimizer-agnostic ``remove_gaussians`` while tracking (camera-only) and the
        optimizer-aware ``prune_points`` while mapping. Returns the number pruned.
        """
        if gaussian_model.get_size() == 0:
            return 0
        ids = torch.as_tensor(list(frame_ids), dtype=torch.long,
                              device=gaussian_model._camera_id.device)
        prune_mask = torch.isin(gaussian_model._camera_id, ids)
        if self._is_tracking_only(gaussian_model):
            return gaussian_model.remove_gaussians(prune_mask)
        gaussian_model.prune_points(prune_mask)
        return int(prune_mask.sum().item())

    @torch.no_grad()
    def _prune_gaussian_params_by_frame_ids(self, gaussian_params: dict, frame_ids) -> int:
        """ In-place pruning of a captured (detached) gaussian-params dict: drops every gaussian
        whose ``camera_id`` is in ``frame_ids``. Returns the number pruned.
        """
        camera_id = gaussian_params["camera_id"]
        ids = torch.as_tensor(list(frame_ids), dtype=torch.long, device=camera_id.device)
        keep = ~torch.isin(camera_id, ids)
        n_pruned = int((~keep).sum().item())
        if n_pruned == 0:
            return 0
        for key in ("xyz", "features_dc", "features_rest", "scaling", "rotation",
                    "opacity", "camera_id", "max_radii2D", "xyz_gradient_accum", "denom"):
            if key in gaussian_params and gaussian_params[key] is not None:
                gaussian_params[key] = gaussian_params[key][keep]
        return n_pruned

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started once a fixed number of keyframes
        have been accumulated since the last submap.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.config["dataset_name"] == "replica":
            # Original submap division: motion heuristic or fixed frame IDs.
            if self.submap_using_motion_heuristic:
                if exceeds_motion_thresholds(
                    self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=50, trans_thre=0.5):
                    return True
            elif frame_id in self.new_submap_frame_ids:
                return True
            return False
        return self.video.counter.value - self.last_submap_kf_idx >= self.new_submap_every_kf

    def _frame_id_to_kf_idx(self, frame_id: int):
        """ Maps a frame id to the keyframe index in the depth video whose timestamp equals it.
        Returns ``None`` if the frame's pose has been dropped by the factor graph (frontend
        keyframe culling / bundle adjustment).
        """
        n_kf = self.video.counter.value
        tstamps = self.video.tstamp[:n_kf]
        matches = (tstamps == frame_id).nonzero(as_tuple=True)[0]

        if matches.numel() == 0:
            return None
        return int(matches[-1])

    def _get_c2w_by_frame_id(self, frame_id: int):
        """ Looks up the camera-to-world pose of the keyframe with the given frame id in the
        depth video. Returns ``None`` if the frame's pose has been dropped by the factor graph.
        """
        kf_idx = self._frame_id_to_kf_idx(frame_id)
        if kf_idx is None:
            return None
        return self.video.get_pose(kf_idx, 'cpu').detach()

    @torch.no_grad()
    def _adjust_saved_submaps_after_ba(self) -> None:
        """ After global BA updates ``video.poses``, applies the resulting pose delta to every
        saved (frozen) submap's gaussians, then advances each submap's stored pose snapshot so
        the next BA computes an incremental delta. Finally persists the adjusted gaussians and
        poses back to the submap checkpoints so reconstruction uses consistent data.
        """
        for submap in self.saved_submaps:
            gaussian_params = submap["gaussian_params"]
            camera_id = gaussian_params["camera_id"]
            keyframe_poses = submap["keyframe_poses"]
            gone_frame_ids = []
            for frame_id, old_c2w in list(keyframe_poses.items()):
                new_c2w = self._get_c2w_by_frame_id(frame_id)
                if new_c2w is None:
                    gone_frame_ids.append(frame_id)
                    continue
                new_c2w = new_c2w.float()   # [4, 4] cpu
                delta = new_c2w @ torch.linalg.inv(old_c2w)
                self._apply_delta_to_xyz_rot(
                    gaussian_params["xyz"], gaussian_params["rotation"],
                    camera_id, frame_id, delta)
                keyframe_poses[frame_id] = new_c2w

            if gone_frame_ids:
                self._prune_gaussian_params_by_frame_ids(gaussian_params, gone_frame_ids)
                for frame_id in gone_frame_ids:
                    keyframe_poses.pop(frame_id, None)
                gone_set = set(gone_frame_ids)
                submap["submap_keyframes"] = [kf for kf in submap["submap_keyframes"]
                                              if kf not in gone_set]

            # Persist the BA-adjusted gaussians and pose snapshot back to disk so the
            # reconstruction evaluator reloads geometry aligned with the final video poses.
            save_dict_to_ckpt(
                {
                    "gaussian_params": gaussian_params,
                    "submap_keyframes": submap["submap_keyframes"],
                    "keyframe_poses": keyframe_poses,
                },
                f"{submap['submap_ckpt_name']}.ckpt",
                directory=self.output_path / "submaps")

    def _save_current_submap(self, gaussian_model: GaussianModel) -> None:
        """ Freezes the current submap: snapshots its gaussians and keyframe poses, saves a
        checkpoint, and registers it in ``saved_submaps`` for later global-BA adjustment.
        """
        gaussian_params = gaussian_model.capture_dict()
        submap_keyframes = sorted(list(self.keyframes_info.keys()))

        # Snapshot each keyframe's camera-to-world pose at save time; the frozen gaussians are
        # anchored to these poses, and this snapshot is the reference for later global-BA deltas.
        # Skip keyframes whose pose was already dropped by the factor graph.
        keyframe_poses = {}
        for kf_id in submap_keyframes:
            c2w = self._get_c2w_by_frame_id(kf_id)
            if c2w is not None:
                keyframe_poses[kf_id] = c2w.detach().clone()

        submap_ckpt_name = str(self.submap_id).zfill(6)
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": submap_keyframes,
            "keyframe_poses": keyframe_poses
        }
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")

        # Keep the frozen submap in memory so global BA can later adjust its gaussians.
        self.saved_submaps.append({
            "gaussian_params": gaussian_params,
            "keyframe_poses": keyframe_poses,
            "submap_keyframes": submap_keyframes,
            "submap_ckpt_name": submap_ckpt_name,
        })

        self.submap_id += 1

    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> GaussianModel:
        """ Saves the current submap's checkpoint and returns a fresh, reset Gaussian model.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        if self.config["dataset_name"] == "replica":
            # Original submap save: snapshot gaussians + keyframes, then reset the model.
            gaussian_params = gaussian_model.capture_dict()
            submap_ckpt_name = str(self.submap_id).zfill(6)
            submap_ckpt = {
                "gaussian_params": gaussian_params,
                "submap_keyframes": sorted(list(self.keyframes_info.keys()))
            }
            save_dict_to_ckpt(
                submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
            gaussian_model = GaussianModel(0)
            gaussian_model.training_setup(self.opt)
            self.mapper.keyframes = []
            self.keyframes_info = {}
            if self.submap_using_motion_heuristic:
                self.new_submap_frame_ids.append(frame_id)
                self.mapping_frame_ids.append(frame_id)
            self.submap_id += 1
            return gaussian_model

        self._save_current_submap(gaussian_model)

        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        self.last_submap_kf_idx = self.video.counter.value
        return gaussian_model


    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

        for frame_id in tqdm(range(len(self.dataset)), desc="Processing frames"):


            with torch.no_grad():
                outsize = (self.H_out_with_edge, self.W_out_with_edge)

                tstamp, image, depth, pose = self.dataset[frame_id]

                image = cv2.resize(image, (self.W_out_with_edge, self.H_out_with_edge))
                image = torch.from_numpy(image).permute(2, 0, 1).float()  # [3, 384, 512]
                image = image.unsqueeze(0)  # [1, 3, 384, 512]

                depth = torch.from_numpy(depth).float()
                depth = F.interpolate(depth[None, None], outsize, mode='nearest')[0, 0]
                if self.W_edge > 0:
                    image = image[:, :, :, self.W_edge:-self.W_edge]
                    depth = depth[:, self.W_edge:-self.W_edge]
                    image = image[:, :, self.H_edge:-self.H_edge, :]
                    depth = depth[self.H_edge:-self.H_edge, :]

                pose = transform_to_quaternion(torch.from_numpy(pose).cuda())

                starting_count = self.video.counter.value
                self.motion_filter.track(None, tstamp, image, depth, pose, self.intrinsic)
                is_new_kf = self.video.counter.value > starting_count

            if is_new_kf:
                kf_idx = self.video.counter.value - 1
                if frame_id in [0, 1]:
                    estimated_c2w = None
                elif self.video.counter.value <= self.frontend.warmup + 1:
                    estimated_c2w = None
                else:
                    prev_c2ws = np.stack([
                        torch2np(self.video.get_pose(0, 'cpu')),
                        torch2np(self.video.get_pose(max(0, kf_idx - 2), 'cpu')),
                        torch2np(self.video.get_pose(kf_idx - 1, 'cpu'))])
                    estimated_c2w = self.tracker.track(frame_id, gaussian_model, prev_c2ws)
                    self.video.poses[kf_idx] = transform_to_quaternion(
                        torch.from_numpy(np.linalg.inv(estimated_c2w)).cuda())

            with torch.no_grad():
                # local bundle adjustment
                self.frontend()
                kf_idx_after = self._frame_id_to_kf_idx(frame_id)

                if kf_idx_after is None:
                    print(
                        f"[CULL] frame {frame_id} was removed by frontend, "
                        f"skip Gaussian mapping."
                    )
                    is_new_kf = False
                else:
                    print(
                        f"[KEEP] frame {frame_id} -> video kf_idx {kf_idx_after}"
                    )

                    curr_kf_idx = self.video.counter.value - 1

            if curr_kf_idx != prev_kf_idx and self.frontend.is_initialized:
                if self.video.counter.value != self.frontend.warmup:
                    if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                        # run online global BA every {self.ba_freq} keyframes
                        print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {tstamp}")
                        self.online_ba.dense_ba(2)
                        prev_ba_idx = curr_kf_idx

            prev_kf_idx = curr_kf_idx

            # Sync the DROID-optimized poses into the mapper's keyframes.
            if is_new_kf:
                self._sync_mapper_poses(gaussian_model)

            torch.cuda.empty_cache()
            #self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Reinitialize gaussian model for new segment
            if not self.only_tracking and self.should_start_new_submap(frame_id):
            #    save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)

            if not self.only_tracking and is_new_kf and frame_id != 0 and self.video.counter.value > self.frontend.warmup:
                kf_idx = self.video.counter.value - 1

                print("\nMapping frame", frame_id)
                gaussian_model.training_setup(self.opt)
                estimate_c2w = torch2np(self.video.get_pose(kf_idx, 'cpu'))
                new_submap = gaussian_model.get_size() == 0
                opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model, new_submap)

                # Keyframes info update
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id,
                    "opt_dict": opt_dict
                }

        print("Final Global BA Triggered!")

        self.ba = Backend(self.droid_net, self.video, self.config)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        if not self.only_tracking and self.config["dataset_name"] != "replica":
            self._sync_mapper_poses(gaussian_model)
            self._save_current_submap(gaussian_model)
            self._adjust_saved_submaps_after_ba()
        print("Final Global BA Done!")

        full_traj_eval(self.traj_filler,
                       f"{self.output_path}/traj",
                       "full_traj",
                       self.dataset, None)

        self.video.save_video(f"{self.output_path}/video.npz")
        
        
    def run_frame_to_model(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

        for frame_id in tqdm(range(len(self.dataset)), desc="Processing frames"):


            #if frame_id in [0, 1]:
            #    estimated_c2w = self.dataset[frame_id][-1]
            #else:
            #    estimated_c2w = self.tracker.track(
           #         frame_id, gaussian_model,
            #        torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))

            #if (frame_id in self.mapping_frame_ids and self.submap_id > 0 and self.config["dataset_name"] == "replica"):
            #    print('\nstart ba')
           #     print("\nframe_id", frame_id)
            #    keyframe = {"id": frame_id, "estimated_c2w": estimated_c2w}
            #    estimated_c2w = self.ba.ba(keyframe, gaussian_model, torch2np(self.estimated_c2ws[frame_id - 1]),
            #                               torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))


            if self.config["dataset_name"] == "replica":
                # ============ replica branch: original Gaussian-SLAM tracking ============
                if frame_id in [0, 1]:
                    estimated_c2w = self.dataset[frame_id][-1]
                else:
                    estimated_c2w = self.tracker.track(
                        frame_id, gaussian_model,
                        torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))

                if (frame_id in self.mapping_frame_ids and self.config["dataset_name"] == "replica") and frame_id > 5:
                    print('\nstart ba')
                    print("\nframe_id", frame_id)
                    keyframe = {"id": frame_id, "estimated_c2w": estimated_c2w}
                    estimated_c2w = self.ba.ba(keyframe, gaussian_model, torch2np(self.estimated_c2ws[frame_id - 1]),
                                               torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))

                self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

                # Reinitialize gaussian model for a new submap (submap division still applies).
                if self.should_start_new_submap(frame_id):
                    gaussian_model = self.start_new_submap(frame_id, gaussian_model)

                # Directly map without any DROID optimization.
                if frame_id in self.mapping_frame_ids:
                    print("\nMapping frame", frame_id)
                    gaussian_model.training_setup(self.opt)
                    estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                    new_submap = not bool(self.keyframes_info)
                    opt_dict = self.mapper.map(frame_id, estimate_c2w, gaussian_model, new_submap)
                    self.keyframes_info[frame_id] = {
                        "keyframe_id": frame_id,
                        "opt_dict": opt_dict
                    }

            if self.config["dataset_name"] == "replica":
            # replica runs the original Gaussian tracker (no DROID), so self.video stays empty;
            # persist the estimated poses in the same {poses, timestamps} format other datasets
            # write to video.npz so the evaluator can read them identically.
                np.savez(
                    f"{self.output_path}/video.npz",
                    poses=torch2np(self.estimated_c2ws),
                    timestamps=np.arange(len(self.dataset), dtype=np.float64),
                )


