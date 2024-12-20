# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import sys

import gym
import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi, gymutil


# Base class for RL tasks
class BaseTask(gym.Env):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        if isinstance(physics_engine, str) and physics_engine == "SIM_PHYSX":
            physics_engine = gymapi.SIM_PHYSX

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = self.sim_device_id

        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        self.num_train_envs = cfg.env.num_envs
        self.num_envs = cfg.env.num_envs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            self.num_envs, self.num_obs, device=self.device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.rew_buf_neg = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.privileged_obs_buf = torch.zeros(
            self.num_envs,
            self.num_privileged_obs,
            device=self.device,
            dtype=torch.float,
        )
        # self.num_privileged_obs = self.num_obs

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "W")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "A")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "S")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "D")
            self.user_inputs = np.zeros(2)

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render_gui(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            user_input_flag = False
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "W" and evt.value > 0:
                    user_input_flag = True
                    self.user_inputs[1] += 0.2
                    self.user_inputs[1] = min(1, self.user_inputs[1])
                elif evt.action == "A" and evt.value > 0:
                    user_input_flag = True
                    self.user_inputs[0] += -0.2
                    self.user_inputs[0] = max(-1, self.user_inputs[0])
                elif evt.action == "S" and evt.value > 0:
                    user_input_flag = True
                    self.user_inputs[1] += -0.2
                    self.user_inputs[1] = max(-1, self.user_inputs[1])
                elif evt.action == "D" and evt.value > 0:
                    user_input_flag = True
                    self.user_inputs[0] += 0.2
                    self.user_inputs[0] = min(1, self.user_inputs[0])
            if not user_input_flag:
                if abs(self.user_inputs[0]) > 0.01:
                    self.user_inputs[0] += -np.sign(self.user_inputs[0]) * 0.01
                else:
                    self.user_inputs[0] = 0
                if abs(self.user_inputs[1]) > 0.01:
                    self.user_inputs[1] += -np.sign(self.user_inputs[1]) * 0.01
                else:
                    self.user_inputs[1] = 0

            # fetch results
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def close(self):
        if self.headless == False:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
