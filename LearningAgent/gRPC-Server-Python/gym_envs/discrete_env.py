from gym_envs.continuous_env import ContinuousPenaltyEnv
import numpy as np
from service_pb2 import GameModeType, ServerParam, PlayerType, PlayerParam, Side, State, TrainerAction, TrainerActions, Vector2D, WorldModel
import service_pb2 as pb2
from service_pb2 import Body_KickOneStep
import gymnasium as gym

class DiscretePenaltyEnv(ContinuousPenaltyEnv):
    def __init__(self, verbose = False):
        self.ANGLE_DIVS = 12
        self.POSSIBLE_KICK_VELS = [.15,.25,.4,.6,1] # relative to ball speed max
        
        super().__init__(verbose)
        self.ANGLE_STEP = 360/self.ANGLE_DIVS
        self.action_space = gym.spaces.Discrete(self.ANGLE_DIVS * len(self.POSSIBLE_KICK_VELS))

    def gym_action_to_soccer_action(self, action, wm: WorldModel):
        angle = action // len(self.POSSIBLE_KICK_VELS) * self.ANGLE_STEP
        normalized_angle = (angle - 180)/180
        kick_vel = self.POSSIBLE_KICK_VELS[action % len(self.POSSIBLE_KICK_VELS)]
        action = [normalized_angle, kick_vel]
        return super().gym_action_to_soccer_action(action, wm)
    def calculate_reward(self, old_observation: State, action: np.ndarray, observation: State) -> float:
        return super().calculate_reward(old_observation, action, observation)/10