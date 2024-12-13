import glob
import os
from typing import Optional, Tuple

import cv2
import groundingdino
import imageio
import isaacgym
import numpy as np
import open3d as o3d
import plotly.express as px
import plotly.graph_objects as go
import torch
import torchvision
import torchvision.transforms as T
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm

from dribblebot.envs import *
from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from XMem.dataset.range_transform import im_normalization
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.inference.inference_core import InferenceCore
from XMem.model.network import XMem


def load_policy(logdir):
    body = torch.jit.load(logdir + "/body.jit", map_location="cpu")

    adaptation_module = torch.jit.load(
        logdir + "/adaptation_module.jit", map_location="cpu"
    )

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_env(label, headless=False):
    dirs = glob.glob(f"runs/{label}/*")
    logdir = sorted(dirs)[-1]

    import yaml

    with open(logdir + "/config.yaml", "rb") as file:
        cfg = yaml.safe_load(file)
        cfg = cfg["Cfg"]

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_tile_roughness = True
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.03]
    # Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    # Cfg.env.num_observations = 75
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.num_border_boxes = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.robot.name = "go1"
    Cfg.sensors.sensor_names = [
        "ObjectSensor",
        "OrientationSensor",
        "RCSensor",
        "JointPositionSensor",
        "JointVelocitySensor",
        "ActionSensor",
        "ActionSensor",
        "ClockSensor",
        "YawSensor",
        "TimingSensor",
    ]
    Cfg.sensors.sensor_args = {
        "ObjectSensor": {},
        "OrientationSensor": {},
        "RCSensor": {},
        "JointPositionSensor": {},
        "JointVelocitySensor": {},
        "ActionSensor": {"delay": 1},
        "ClockSensor": {},
        "YawSensor": {},
        "TimingSensor": {},
    }

    Cfg.sensors.privileged_sensor_names = {
        "BodyVelocitySensor": {},
        "ObjectVelocitySensor": {},
    }
    Cfg.sensors.privileged_sensor_args = {
        "BodyVelocitySensor": {},
        "ObjectVelocitySensor": {},
    }
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"
    Cfg.env.num_privileged_obs = 6

    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    policy = load_policy(logdir)
    return env, policy


def depth2fgpcd(
    depth: torch.Tensor,
    mask: torch.Tensor,
    cam_params: torch.Tensor,
    device: str = "cuda:0",
    preserve_zero: bool = False,
) -> torch.Tensor:
    # depth: (h, w)
    # mask: (h, w)
    # cam_params: (4,)
    if not preserve_zero:
        mask = torch.logical_and(mask, depth < 0)
    fgpcd = torch.zeros((int(mask.sum()), 3), device=device)
    fx, fy, cx, cy = cam_params
    pos_y, pos_x = torch.meshgrid(
        torch.arange(depth.shape[0], device=device),
        torch.arange(depth.shape[1], device=device),
        indexing="ij",
    )
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]

    return fgpcd


def voxel_downsample(
    points: torch.Tensor, voxel_size: float, points_color: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    # points: (n, 3)
    # voxel_size: float
    # points_color: (n, 3)
    points = points.float()
    voxel_indices = (points / voxel_size).floor()
    unique_voxels, indices = torch.unique(voxel_indices, return_inverse=True, dim=0)
    voxel_points = torch.zeros(
        unique_voxels.size(0), points.size(1), dtype=torch.float, device=points.device
    )
    voxel_points.index_add_(0, indices, points)
    counts = torch.zeros(
        unique_voxels.size(0), 1, dtype=torch.float, device=points.device
    )
    counts.index_add_(0, indices, torch.ones(points.size(0), 1, device=points.device))
    centroids = voxel_points / counts
    if points_color is not None:
        points_color = points_color.float()
        voxel_colors = torch.zeros(
            unique_voxels.size(0),
            points_color.size(1),
            dtype=torch.float,
            device=points.device,
        )
        voxel_colors.index_add_(0, indices, points_color)
        centroids_color = voxel_colors / counts
    else:
        centroids_color = torch.Tensor()

    return centroids, centroids_color


def multi_camera_2_pcd(
    colors: torch.Tensor,
    depths: torch.Tensor,
    Ks: torch.Tensor,
    poses: torch.Tensor,
    depth_max: Optional[float] = None,
    downsample: bool = True,
    downsample_r: float = 0.01,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    # colors: (n, h, w, 3)
    # depths: (n, h, w)
    # Ks: (n, 3, 3)
    # poses: (n, 4, 4)
    N, H, W, _ = colors.shape
    colors = colors / 255.0
    start = 0
    end = N
    step = 1
    pcds = torch.empty((0, 3), device=device)
    pcd_colors = torch.empty((0, 3), device=device)
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        K = Ks[i]
        cam_params = torch.tensor(
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], device=device
        )  # fx, fy, cx, cy
        mask = depth < 0
        if depth_max is not None:
            mask &= depth > -depth_max
        pcd = depth2fgpcd(depth, mask, cam_params, device)

        pose = poses[i]
        pose = torch.inverse(pose)

        trans_pcd = pose @ torch.cat(
            (pcd.T, torch.ones((1, pcd.shape[0]), device=device)), axis=0
        )
        trans_pcd = trans_pcd[:3].T

        color = color[mask]

        pcds = torch.cat((pcds, trans_pcd), axis=0)
        pcd_colors = torch.cat((pcd_colors, color), axis=0)

    if downsample:
        pcds, pcd_colors = voxel_downsample(pcds, downsample_r, pcd_colors)

    return pcds, pcd_colors


def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    if pcd.shape[0] == 0:
        return pcd_o3d
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        color = color.reshape(-1, 3)
        color = color - color.min()
        color = color / color.max()
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def sphere_fit(
    pts: np.array,
    thresh: float = 0.2,
    maxIteration: int = 1000,
    radius: Optional[float] = None,
):
    n_points = pts.shape[0]
    best_inliers = []
    best_center = []
    best_radius = 0

    for it in range(maxIteration):
        # Samples 4 random points
        id_samples = np.random.choice(n_points, 4, replace=False)
        pt_samples = pts[id_samples]

        # We calculate the 4x4 determinant by dividing the problem in determinants of 3x3 matrix

        # Multiplied by (x²+y²+z²)
        d_matrix = np.ones((4, 4))
        for i in range(4):
            d_matrix[i, 0] = pt_samples[i, 0]
            d_matrix[i, 1] = pt_samples[i, 1]
            d_matrix[i, 2] = pt_samples[i, 2]
        M11 = np.linalg.det(d_matrix)

        # Multiplied by x
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 1]
            d_matrix[i, 2] = pt_samples[i, 2]
        M12 = np.linalg.det(d_matrix)

        # Multiplied by y
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 2]
        M13 = np.linalg.det(d_matrix)

        # Multiplied by z
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 1]
        M14 = np.linalg.det(d_matrix)

        # Multiplied by 1
        for i in range(4):
            d_matrix[i, 0] = np.dot(pt_samples[i], pt_samples[i])
            d_matrix[i, 1] = pt_samples[i, 0]
            d_matrix[i, 2] = pt_samples[i, 1]
            d_matrix[i, 3] = pt_samples[i, 2]
        M15 = np.linalg.det(d_matrix)

        # Now we calculate the center and radius
        center = [0.5 * (M12 / M11), -0.5 * (M13 / M11), 0.5 * (M14 / M11)]
        if radius is None:
            radius = np.sqrt(np.dot(center, center) - (M15 / M11))

        # Distance from a point
        pt_id_inliers = []  # list of inliers ids
        dist_pt = center - pts
        dist_pt = np.linalg.norm(dist_pt, axis=1)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt - radius) <= thresh)[0]

        if len(pt_id_inliers) > len(best_inliers):
            best_inliers = pt_id_inliers
            best_center = center
            best_radius = radius

    return best_center, best_radius, best_inliers


def segment(
    sam_predictor: SamPredictor,
    # sam_predictor: SAM2ImagePredictor,
    image: np.ndarray,
    xyxy: np.ndarray,
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def detect_and_seg(
    image,
    text_prompts,
    dino_model: GroundingDINOModel,
    sam_model: SamPredictor,
):
    # :param image: [H, W, 3] BGR
    assert len(image.shape) == 3

    if len(text_prompts) == 0:
        labels = ["background"]
        # add detections mask for background
        bg_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
        mask = np.expand_dims(bg_mask, axis=0)
        bg_conf = 1.0
        scores = np.array([bg_conf])

        return mask, labels, scores, [[0, 0, 0, 0]]

    BOX_THRESHOLD = 0.7
    TEXT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.8

    detections = dino_model.predict_with_classes(
        image=image,
        classes=text_prompts,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # non-maximum suppression
    nms_idx = (
        torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD,
        )
        .numpy()
        .tolist()
    )
    detections.bboxes = detections.xyxy[nms_idx]
    detections.scores = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # remove large boxes that include small boxes
    # remove_idx = []
    # for i in range(len(detections.bboxes)):
    #     for j in range(i + 1, len(detections.bboxes)):
    #         compare_flag_0 = detections.bboxes[i][0] > detections.bboxes[j][0]
    #         compare_flag_1 = detections.bboxes[i][1] > detections.bboxes[j][1]
    #         compare_flag_2 = detections.bboxes[i][2] > detections.bboxes[j][2]
    #         compare_flag_3 = detections.bboxes[i][3] > detections.bboxes[j][3]
    #         if (
    #             not compare_flag_0
    #             and not compare_flag_1
    #             and compare_flag_2
    #             and compare_flag_3
    #         ):
    #             remove_idx.append(i)
    #         if (
    #             compare_flag_0
    #             and compare_flag_1
    #             and not compare_flag_2
    #             and not compare_flag_3
    #         ):
    #             remove_idx.append(j)
    # detections.bboxes = np.delete(detections.bboxes, remove_idx, axis=0)
    # detections.scores = np.delete(detections.scores, remove_idx, axis=0)
    # detections.class_id = np.delete(detections.class_id, remove_idx, axis=0)

    # convert detections to masks
    mask = segment(
        sam_predictor=sam_model,
        image=image,
        xyxy=detections.bboxes,
    )
    mask = mask.astype(bool)

    # remove large masks that include small masks
    # remove_idx = []
    # filter_thresh = 100
    # for i in range(len(mask)):
    #     for j in range(i + 1, len(mask)):
    #         IoU, overlap, only_a, only_b = maskIoU(mask[i], mask[j])
    #         if overlap > filter_thresh:
    #             remove_which = -1
    #             if only_a + only_b < filter_thresh:
    #                 if only_a > only_b:
    #                     remove_which = j
    #                 else:
    #                     remove_which = i
    #             elif only_a < filter_thresh:
    #                 remove_which = j
    #             elif only_b < filter_thresh:
    #                 remove_which = i
    #             if remove_which != -1:
    #                 remove_idx.append(remove_which)
    # mask = np.delete(mask, remove_idx, axis=0)
    # detections.bboxes = np.delete(detections.bboxes, remove_idx, axis=0)
    # detections.scores = np.delete(detections.scores, remove_idx, axis=0)
    # detections.class_id = np.delete(detections.class_id, remove_idx, axis=0)

    labels = ["background"]
    for query_i in detections.class_id.tolist():
        if query_i is None:
            continue
        else:
            labels.append(text_prompts[query_i])

    # add detections mask for background
    if mask.shape[0] == 0:
        bg_mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
        mask = np.expand_dims(bg_mask, axis=0)
    else:
        bg_mask = ~np.bitwise_or.reduce(mask, axis=0)
        mask = np.concatenate([np.expand_dims(bg_mask, axis=0), mask], axis=0)
    bg_conf = 1.0
    scores = np.concatenate([np.array([bg_conf]), np.array(detections.scores)], axis=0)

    bboxes = np.concatenate([np.array([[0, 0, 0, 0]]), detections.bboxes], axis=0)
    return mask, labels, scores, bboxes


def play_go1(headless=True):
    curr_path = os.path.dirname(os.path.abspath(__file__))

    label = "dribbling/bvggoq26"
    env, policy = load_env(label, headless=headless)

    # load Grounding DINO model
    gdino_config = os.path.join(
        groundingdino.__path__[0], "config/GroundingDINO_SwinB_cfg.py"
    )
    gdino_ckpt_path = os.path.join(curr_path, "foundational_models/ckpts")
    gdino_ckpt_name = "groundingdino_swinb_cogcoor.pth"
    gdino_ckpt = f"{gdino_ckpt_path}/{gdino_ckpt_name}"
    if not os.path.exists(gdino_ckpt):
        print("Downloading Grounding DINO model...")
        os.system(
            f"wget -P {gdino_ckpt_path} https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/{gdino_ckpt_name}"
        )
    ground_dino_model = GroundingDINOModel(gdino_config, gdino_ckpt, device=env.device)

    # load SAM model
    sam_ckpt_path = os.path.join(curr_path, "foundational_models/ckpts")
    sam_ckpt_name = "sam_vit_h_4b8939.pth"
    sam_ckpt = f"{sam_ckpt_path}/{sam_ckpt_name}"
    sam_model_type = "vit_h"
    if not os.path.exists(sam_ckpt):
        print("Downloading SAM model...")
        os.system(
            f"wget -P {sam_ckpt_path} https://dl.fbaipublicfiles.com/segment_anything/{sam_ckpt_name}"
        )
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    sam.to(device=env.device)
    sam_model = SamPredictor(sam)
    sam_mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64)

    # load XMem model
    xmem_ckpt_path = os.path.join(curr_path, "XMem/saves")
    xmem_ckpt_name = "XMem.pth"
    xmem_ckpt = f"{xmem_ckpt_path}/{xmem_ckpt_name}"
    if not os.path.exists(xmem_ckpt):
        print("Downloading XMem model...")
        os.system(
            f"wget -P {xmem_ckpt_path} https://github.com/hkchengrex/XMem/releases/download/v1.0/{xmem_ckpt_name}"
        )
    xmem_config = {
        "model": xmem_ckpt,
        "disable_long_term": False,
        "enable_long_term": True,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "save_scores": False,
        "size": 480,
        "key_dim": 64,
        "value_dim": 512,
        "hidden_dim": 64,
        "enable_long_term_count_usage": True,
    }

    network = XMem(xmem_config, xmem_config["model"]).to(env.device).eval()
    model_weights = torch.load(xmem_config["model"])
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    xmem_mapper = MaskMapper()
    xmem_processors = [InferenceCore(network, config=xmem_config) for _ in range(2)]
    if xmem_config["size"] < 0:
        xmem_im_transform = T.Compose(
            [
                T.ToTensor(),
                im_normalization,
            ]
        )
        xmem_mask_transform = None
    else:
        xmem_im_transform = T.Compose(
            [
                T.ToTensor(),
                im_normalization,
                T.Resize(
                    xmem_config["size"], interpolation=T.InterpolationMode.BILINEAR
                ),
            ]
        )
        xmem_mask_transform = T.Compose(
            [
                T.Resize(
                    xmem_config["size"], interpolation=T.InterpolationMode.NEAREST
                ),
            ]
        )
    xmem_first_mask_loaded = False
    track_ids = [0]

    num_eval_steps = 50000
    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
    }

    body_height_cmd = 0.0
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.09
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    measured_y_vels = np.zeros(num_eval_steps)
    target_x_vels = np.zeros(num_eval_steps)
    target_y_vels = np.zeros(num_eval_steps)
    joint_positions = np.zeros((num_eval_steps, 12))

    mp4_writer = imageio.get_writer(
        "outputs/dribbling.mp4", fps=50, macro_block_size=None
    )
    front_rgb_recorder = imageio.get_writer(
        "outputs/front_rgb.mp4", fps=50, macro_block_size=None
    )
    front_depth_recorder = imageio.get_writer(
        "outputs/front_depth.mp4", fps=50, macro_block_size=None
    )
    bottom_rgb_recorder = imageio.get_writer(
        "outputs/bottom_rgb.mp4", fps=50, macro_block_size=None
    )
    bottom_depth_recorder = imageio.get_writer(
        "outputs/bottom_depth.mp4", fps=50, macro_block_size=None
    )
    depth_max = 1

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window("Point Cloud", width=640 * 2, height=480 * 2, visible=True)
    visualizer.get_render_option().background_color = [0.9, 0.9, 0.9]
    curr_d3fields = o3d.geometry.PointCloud()
    init_flag = True

    front_detect_flag = False
    bottom_detect_flag = False

    obs = env.reset()
    ep_rew = 0
    for i in tqdm(
        range(num_eval_steps), desc="Evaluating", colour="green", ascii=" 123456789>"
    ):
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = env.user_inputs[0], env.user_inputs[1], 0.0
        target_x_vels[i] = x_vel_cmd
        target_y_vels[i] = y_vel_cmd
        with torch.no_grad():
            actions = policy(obs)
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)
        measured_x_vels[i] = env.base_lin_vel[0, 0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        joint_positions[i] = env.dof_pos[0, :].cpu()
        ep_rew += rew

        front_rgb_images = env.get_rgb_images([0])
        front_rgb_image = front_rgb_images["front"][0][..., :3]
        rgb_image = front_rgb_image.cpu().numpy()
        front_rgb_recorder.append_data(rgb_image.astype(np.uint8))
        image = rgb_image[..., ::-1] / 255
        cv2.imshow("front camera rgb", image)
        if not front_detect_flag:
            mask, label, score, bbox = detect_and_seg(
                rgb_image[..., ::-1].astype(np.uint8),
                ["yellow ball"],
                ground_dino_model,
                sam_model,
            )
            if "yellow ball" in label:
                front_detect_flag = True
                annotate_plotly(
                    rgb_image[..., ::-1].astype(np.uint8),
                    label,
                    score,
                    bbox,
                    mask,
                    get_class_color_dict(),
                )

        front_depth_images = env.get_depth_images([0])
        front_depth_image = front_depth_images["front"][0]
        front_depth_image[front_depth_image == -np.inf] = 0
        front_depth_image[front_depth_image < -depth_max] = -depth_max
        depth_image = -front_depth_image.cpu().numpy()
        depth_image = 1 - depth_image / depth_max
        depth_image = depth_image.clip(0, 1)
        cv2.imshow("front camera depth", depth_image)
        depth_image = depth_image * 255
        front_depth_recorder.append_data(depth_image.astype(np.uint8))

        bottom_rgb_images = env.get_rgb_images([0])
        bottom_rgb_image = bottom_rgb_images["bottom"][0][..., :3]
        rgb_image = bottom_rgb_image.cpu().numpy()
        bottom_rgb_recorder.append_data(rgb_image.astype(np.uint8))
        image = rgb_image[..., ::-1] / 255
        cv2.imshow("bottom camera rgb", image)
        if not bottom_detect_flag:
            mask, label, score, bbox = detect_and_seg(
                rgb_image[..., ::-1].astype(np.uint8),
                ["yellow ball"],
                ground_dino_model,
                sam_model,
            )
            if "yellow ball" in label:
                bottom_detect_flag = True
                annotate_plotly(
                    rgb_image[..., ::-1].astype(np.uint8),
                    label,
                    score,
                    bbox,
                    mask,
                    get_class_color_dict(),
                )

        bottom_depth_images = env.get_depth_images([0])
        bottom_depth_image = bottom_depth_images["bottom"][0]
        bottom_depth_image[bottom_depth_image == -np.inf] = 0
        bottom_depth_image[bottom_depth_image < -depth_max] = -depth_max
        depth_image = -bottom_depth_image.cpu().numpy()
        depth_image = 1 - depth_image / depth_max
        depth_image = depth_image.clip(0, 1)
        cv2.imshow("bottom camera depth", depth_image)
        depth_image = depth_image * 255
        bottom_depth_recorder.append_data(depth_image.astype(np.uint8))

        colors = torch.stack((front_rgb_image, bottom_rgb_image))
        depths = torch.stack((front_depth_image, bottom_depth_image))

        front_intr = env.get_camera_intrinsics(0, "front")
        front_extr = env.get_camera_extrinsics(0, "front")
        bottom_intr = env.get_camera_intrinsics(0, "bottom")
        bottom_extr = env.get_camera_extrinsics(0, "bottom")
        intrs = torch.stack((front_intr, bottom_intr))
        extrs = torch.stack((front_extr, bottom_extr))

        pcd, pcd_color = multi_camera_2_pcd(
            colors, depths, intrs, extrs, depth_max=depth_max
        )
        if pcd.shape[0]:
            red_channel = pcd_color[:, 0]
            green_channel = pcd_color[:, 1]
            blue_channel = pcd_color[:, 2]
            filter_r = (0.2 <= red_channel) & (red_channel <= 0.6)
            filter_g = (0.2 <= green_channel) & (green_channel <= 0.6)
            filter_b = (0.0 <= blue_channel) & (blue_channel <= 0.2)
            filter_mask = filter_r & filter_g & filter_b
            sphere_pcd = pcd[filter_mask].cpu().numpy()
            # pcd = pcd[filter_mask]
            # pcd_color = pcd_color[filter_mask]

            pcd_o3d = np2o3d(pcd.cpu().numpy(), pcd_color.cpu().numpy())
            sphere_center, sphere_radius, sphere_inliers = sphere_fit(
                sphere_pcd,
                thresh=0.005,
                maxIteration=100,
                radius=Cfg.ball.radius,
            )
            if init_flag:
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
                visualizer.add_geometry(origin)

                curr_d3fields.points = pcd_o3d.points
                curr_d3fields.colors = pcd_o3d.colors
                visualizer.add_geometry(curr_d3fields)

                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=sphere_radius
                )
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color([0.0, 1.0, 0.0])
                mesh_sphere.translate(sphere_center, relative=False)
                visualizer.add_geometry(mesh_sphere)

                # if view_ctrl_info is not None:
                #     view_control = visualizer.get_view_control()
                #     view_control.set_front(view_ctrl_info["front"])
                #     view_control.set_lookat(view_ctrl_info["lookat"])
                #     view_control.set_up(view_ctrl_info["up"])
                #     view_control.set_zoom(view_ctrl_info["zoom"])

                init_flag = False
            else:
                curr_d3fields.points = pcd_o3d.points
                curr_d3fields.colors = pcd_o3d.colors
                visualizer.update_geometry(curr_d3fields)

                mesh_sphere.translate(sphere_center, relative=False)
                visualizer.update_geometry(mesh_sphere)

            visualizer.poll_events()
            visualizer.update_renderer()

        render_image = env.render(mode="rgb_array")
        auto_follow_image = render_image[..., :3]
        auto_follow_image = auto_follow_image[..., ::-1]
        cv2.imshow("auto-follow camera", auto_follow_image)

        canvas_size = 200
        half_canvas_size = canvas_size // 2
        direction_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8)
        direction_canvas = cv2.arrowedLine(
            direction_canvas,
            (half_canvas_size, half_canvas_size),
            (
                half_canvas_size + int(x_vel_cmd * half_canvas_size),
                half_canvas_size - int(y_vel_cmd * half_canvas_size),
            ),
            (0, 255, 0),
            5,
        )
        cv2.imshow("direction", direction_canvas)

        cv2.waitKey(1)
        mp4_writer.append_data(render_image)

    front_rgb_recorder.close()
    front_depth_recorder.close()
    bottom_rgb_recorder.close()
    bottom_depth_recorder.close()
    mp4_writer.close()
    visualizer.close()

    # plot target and measured forward velocity
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(12, 5))

    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        measured_x_vels,
        color="red",
        linestyle="-",
        label="Measured",
    )
    axs[0].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        target_x_vels,
        color="green",
        linestyle="--",
        label="Desired",
    )
    axs[0].legend()
    axs[0].set_title("Linear Velocity Along X")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        measured_y_vels,
        color="blue",
        linestyle="-",
        label="Measured",
    )
    axs[1].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        target_y_vels,
        color="green",
        linestyle="--",
        label="Desired",
    )
    axs[1].legend()
    axs[1].set_title("Linear Velocity Along Y")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Velocity (m/s)")

    axs[2].plot(
        np.linspace(0, num_eval_steps * env.dt, num_eval_steps),
        joint_positions,
        linestyle="-",
        label="Measured",
    )
    axs[2].set_title("Joint Positions")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


def annotate_plotly(
    image,
    label,
    score,
    box,
    mask,
    class_color_dict: Optional[dict] = None,
):
    # If class_colors is not provided, generate random colors for each class
    num_obj = len(label)
    if class_color_dict is None:
        colors = random_named_css_colors(num_obj)
    else:
        colors = {}
        for i in range(num_obj):
            if label[i] == "background" or label[i] == "table":
                continue
            if label[i] not in class_color_dict:
                raise ValueError(f"Class {label[i]} not found in class_color_dict")
            colors[i] = class_color_dict[label[i]]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = px.imshow(image)

    # Add bounding boxes
    shapes = []
    annotations = []
    for i in range(num_obj):
        if label[i] == "background" or label[i] == "table":
            continue
        polygon = mask_to_polygon(mask[i])  # (3, 480, 640)

        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in polygon] + [polygon[0][0]],
                y=[point[1] for point in polygon] + [polygon[0][1]],
                mode="lines",
                line=dict(color=colors[i], width=2),
                fill="toself",
                name=f"{label[i]}: {score[i]:.2f}",
            )
        )

        xmin, ymin, xmax, ymax = [round(bbox) for bbox in box[i]]
        shape = [
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=colors[i]),
            )
        ]
        annotation = [
            dict(
                x=(xmin + xmax) // 2,
                y=(ymin + ymax) // 2,
                xref="x",
                yref="y",
                text=f"{label[i]}: {score[i]:.2f}",
            )
        ]

        shapes.append(shape)
        annotations.append(annotation)

    # Update layout
    button_shapes = [dict(label="None", method="relayout", args=["shapes", []])]
    button_shapes = button_shapes + [
        dict(label=f"Detection {idx + 1}", method="relayout", args=["shapes", shape])
        for idx, shape in enumerate(shapes)
    ]
    button_shapes = button_shapes + [
        dict(label="All", method="relayout", args=["shapes", sum(shapes, [])])
    ]

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        updatemenus=[dict(type="buttons", direction="up", buttons=button_shapes)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Show plot
    fig.show()


def mask_to_polygon(mask: np.ndarray):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon


def random_named_css_colors(num_colors: int):
    """
    Returns a list of randomly selected named CSS colors.

    Args:
    - num_colors (int): Number of random colors to generate.

    Returns:
    - list: List of randomly selected named CSS colors.
    """
    # List of named CSS colors
    named_css_colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    # Sample random named CSS colors
    return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))


def get_class_color_dict():
    class_color_dict = {
        "yellow ball": "green",
        "red soda can": "red",
        "red mug": "red",
        "green soda can": "green",
        "green mug": "green",
        "blue soda can": "blue",
        "blue mug": "blue",
        "brown soda can": "brown",
        "small square wooden pad": "purple",
        "square wooden container": "purple",
        "red pad": "orange",
        "blue pad": "purple",
        "coat rack": "purple",
        "battery": "orange",
        "crate": "purple",
        "mug": "red",
        "mug tree": "purple",
        "book": "green",
        "inst_0": "red",
        "inst_1": "blue",
        "inst_2": "green",
    }
    return class_color_dict


if __name__ == "__main__":
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
