from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from dribblebot.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class ObjectVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        # self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):

        return self.env.object_lin_vel
    
    # Privileged sensor input does not contain noise
    # def get_noise_vec(self):
    #     import torch
    #     return torch.zeros(3, device=self.env.device)

    def get_dim(self):
        return 3