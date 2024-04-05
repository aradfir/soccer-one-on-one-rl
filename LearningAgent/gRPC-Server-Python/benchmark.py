from concurrent import futures
import threading
import time
from stable_baselines3.common.logger import configure
import grpc
from stable_baselines3 import DDPG, DQN
from gym_envs.continuous_env import ContinuousPenaltyEnv, TerminalStates
from gym_envs.discrete_env import DiscretePenaltyEnv
from gym_envs.discrete_manual_angle_discretization import DribbleAndShootAngleDiscretizationEnv
from gym_envs.discrete_with_helios_shoot_env import DiscreteEnvWShoot
from gym_grpc_server import GymGame, serve
from stable_baselines3.common.callbacks import CheckpointCallback
import service_pb2_grpc as pb2_grpc


if __name__ == "__main__":
    # gym_env = DribbleAndShootAngleDiscretizationEnv(verbose=False)
    gym_env = ContinuousPenaltyEnv(verbose=False)
    trainer_started = threading.Event()
    server_thread = threading.Thread(target=serve, args=(gym_env,trainer_started))
    server_thread.daemon = True
    server_thread.start()
    print("Await trainer")
    trainer_started.wait()
    model = DDPG('MlpPolicy', gym_env)
    # model = DQN('MlpPolicy', gym_env, exploration_initial_eps=0.05)
    model = model.load(path="best_models/DDPG_YuShan_Learn.zip",env=gym_env)
    outcomes = {TerminalStates.GOAL:0,TerminalStates.OOB:0,TerminalStates.GOALIE_CATCH:0,TerminalStates.TIMEOUT:-2,TerminalStates.NOT_TERMINAL:0}
    num_ep = 0
    observation, _ = gym_env.reset()
    # note: in penalty first 2 episodes are always timeout for some reason
    while num_ep < 102:
        # get action from the model
        action, _ = model.predict(observation, deterministic=True)
        # action = gym_env.action_space.sample()
        # print(f"Action: {action}")
        # get observation from the environment 
        observation, reward, terminated ,truncated, info = gym_env.step(action)
        # print(f"Observation: {observation}, Reward: {reward}, terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            num_ep += 1
            status = info['terminal_state']
            if status == TerminalStates.GOAL:
                outcomes[TerminalStates.GOAL] += 1
            elif status == TerminalStates.OOB:
                outcomes[TerminalStates.OOB] += 1
            elif status == TerminalStates.GOALIE_CATCH:
                outcomes[TerminalStates.GOALIE_CATCH] += 1
            elif status == TerminalStates.TIMEOUT:
                outcomes[TerminalStates.TIMEOUT] += 1
            elif status == TerminalStates.NOT_TERMINAL:
                outcomes[TerminalStates.NOT_TERMINAL] += 1
            observation, info = gym_env.reset()

    print(f"Outcomes: {outcomes}")
    

