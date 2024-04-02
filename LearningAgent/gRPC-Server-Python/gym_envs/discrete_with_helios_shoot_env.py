
from numpy import ndarray
from gym_envs.discrete_env import DiscretePenaltyEnv
import gymnasium as gym
import service_pb2 as pb2

from service_pb2 import WorldModel

class DiscreteEnvWShoot(DiscretePenaltyEnv):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.action_space = gym.spaces.Discrete(self.ANGLE_DIVS * len(self.POSSIBLE_KICK_VELS) + 1)
    
    def gym_action_to_soccer_action(self, action, wm: WorldModel):
        if action == 0:
            return pb2.PlayerAction(helios_shoot=pb2.HeliosShoot())
        return super().gym_action_to_soccer_action(action - 1, wm)
    
    def calculate_reward(self, old_observation: pb2.State, action: ndarray, observation: pb2.State) -> float:
        shoot_reward = 0
        if action == 0:
            shoot_reward = -10
        return super().calculate_reward(old_observation, action, observation) + shoot_reward
        