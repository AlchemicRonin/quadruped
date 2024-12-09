import glob

import cv2
import imageio
import isaacgym
import numpy as np
import torch
from tqdm import tqdm

from dribblebot.envs import *
from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.go1.velocity_tracking import VelocityTrackingEasyEnv


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
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.05]

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


def play_go1(headless=True):
    label = "dribbling/bvggoq26"
    env, policy = load_env(label, headless=headless)

    num_eval_steps = 5000
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
    depth_max = 2

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

        rgb_images = env.get_rgb_images([0])
        rgb_image = rgb_images["front"][0][..., :3].cpu().numpy()
        front_rgb_recorder.append_data(rgb_image.astype(np.uint8))
        rgb_image = rgb_image[..., ::-1] / 255
        cv2.imshow("front camera rgb", rgb_image)

        depth_images = env.get_depth_images([0])
        depth_image = depth_images["front"][0].cpu().numpy()
        depth_image[depth_image == -np.inf] = 0
        depth_image[depth_image < -depth_max] = -depth_max
        depth_image = 1 + depth_image / depth_max
        depth_image = depth_image.clip(0, 1)
        cv2.imshow("front camera depth", depth_image)
        depth_image = depth_image * 255
        front_depth_recorder.append_data(depth_image.astype(np.uint8))

        rgb_images = env.get_rgb_images([0])
        rgb_image = rgb_images["bottom"][0][..., :3].cpu().numpy()
        bottom_rgb_recorder.append_data(rgb_image.astype(np.uint8))
        rgb_image = rgb_image[..., ::-1] / 255
        cv2.imshow("bottom camera rgb", rgb_image)

        depth_images = env.get_depth_images([0])
        depth_image = depth_images["bottom"][0].cpu().numpy()
        depth_image[depth_image == -np.inf] = 0
        depth_image[depth_image < -depth_max] = -depth_max
        depth_image = 1 + depth_image / depth_max
        depth_image = depth_image.clip(0, 1)
        cv2.imshow("bottom camera depth", depth_image)
        depth_image = depth_image * 255
        bottom_depth_recorder.append_data(depth_image.astype(np.uint8))

        img = env.render(mode="rgb_array")
        auto_follow_image = img[..., :3]
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

        mp4_writer.append_data(img)

        out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)

    front_rgb_recorder.close()
    front_depth_recorder.close()
    bottom_rgb_recorder.close()
    bottom_depth_recorder.close()
    mp4_writer.close()

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


if __name__ == "__main__":
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
