from gym_envs.continuous_env import ContinuousPenaltyEnv
import numpy as np
from service_pb2 import GameModeType, ServerParam, PlayerType, PlayerParam, Side, State, TrainerAction, TrainerActions, Vector2D, WorldModel
import service_pb2 as pb2
from service_pb2 import Body_KickOneStep
import gymnasium as gym

class DiscretePenaltyEnv(ContinuousPenaltyEnv):
    def __init__(self, verbose = False):
        self.ANGLE_DIVS = 12
        self.POSSIBLE_KICK_VELS = [.1,.2,.3,.4,.5,.6,.7,.8] # relative to ball speed max
        
        super().__init__(verbose)
        self.ANGLE_STEP = 360/self.ANGLE_DIVS
        self.action_space = gym.spaces.Discrete(self.ANGLE_DIVS * len(self.POSSIBLE_KICK_VELS))

    def gym_action_to_soccer_action(self, action, wm: WorldModel):
        angle = action // len(self.POSSIBLE_KICK_VELS) * self.ANGLE_STEP
        kick_vel = self.POSSIBLE_KICK_VELS[action % len(self.POSSIBLE_KICK_VELS)] * self.server_param.ball_speed_max
        return Body_KickOneStep(angle, kick_vel)