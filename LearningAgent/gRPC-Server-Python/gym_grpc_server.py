import time

import numpy as np
from grpc_server import Game
import threading
from multiprocessing import Process,Queue
from concurrent import futures
import grpc
from gym_envs.continuous_env import ContinuousPenaltyEnv
from stable_baselines3.common.logger import configure
from gym_envs.discrete_env import DiscretePenaltyEnv
from gym_envs.discrete_with_helios_shoot_env import DiscreteEnvWShoot
from gym_envs.discrete_with_shoot_action import DribbleAndShootEnv
from gym_envs.discrete_manual_angle_discretization import DribbleAndShootAngleDiscretizationEnv
import service_pb2_grpc as pb2_grpc
import service_pb2 as pb2
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C,TD3,DDPG,DQN

from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,EvalCallback
from queue import Empty, Full

DEBUG_GRPC = False
DEBUG_GYM = False
# flag to check if setup is done

class GymGame(Game):
    def __init__(self, gym_env:ContinuousPenaltyEnv, trainer_start_event:threading.Event ):
        super().__init__()
        self.opp_goalie_start = False
        self.was_set_play_before = False
        self.gym_env: ContinuousPenaltyEnv = gym_env
        self.trainer_started = trainer_start_event
    
    def SendServerParams(self, request: pb2.ServerParam, context):
        self.gym_env.server_param = request
        return super().SendServerParams(request, context)
    
    def SendPlayerParams(self, request: pb2.PlayerParam, context):
        self.gym_env.player_param = request
        return super().SendPlayerParams(request, context)
    
    def SendPlayerType(self, request: pb2.PlayerType, context):
        self.gym_env.player_type = request
        return super().SendPlayerType(request, context)
    
    def log(self, msg):
        if DEBUG_GRPC:
            print(msg)
    
    # returns true when all opps are connected
    def wait_for_opponents(self, wm:pb2.WorldModel) -> bool:
        if self.opp_goalie_start:
            return True
        goalie_opp = wm.their_players_dict[1]
        self.log(f"Opp count:{len(wm.opponents)} Goalie Opp:{goalie_opp.uniform_number}")
        if goalie_opp.uniform_number == 1:
            self.opp_goalie_start = True
            return True
        return False
    
    def GetTrainerActions(self, request: pb2.State, context):
        
        if not self.wait_for_opponents(request.world_model):
            return pb2.TrainerActions()
        
        self.trainer_started.set()
        try:
            self.gym_env.trainer_action_queue.get(block=False)
            cycle = request.world_model.cycle
            self.gym_env.episode_start_cycle = cycle + 1
            print(f"Episode Start: {cycle + 1}")
            return self.gym_env.get_trainer_reset_commands()
        except Empty:
            return pb2.TrainerActions()
    
    def GetPlayerActions(self, request, context): 
        self.log(f"***** GOT PLAYER CALL, CYCLE = {request.world_model.cycle}, GAMEMODE = {request.world_model.game_mode_type}")
        # append neck action first to the action list
        actions = pb2.PlayerActions()
        wm = request.world_model
        # self.gym_env.append_intermittent_rewards(wm)
        actions.actions.append(pb2.PlayerAction(neck_turn_to_ball_or_scan=pb2.Neck_TurnToBallOrScan()))
        if not self.wait_for_opponents(request.world_model):
            return actions
        if wm.game_mode_type not in self.gym_env.PLAY_STATES and not self.was_set_play_before:
            self.log("***** IN FIRST SETPLAY, SEND OBS")
            self.was_set_play_before = True
            self.gym_env.clear_actions_queue()
            self.gym_env.observation_queue.put(request)
            return actions
        if self.was_set_play_before and wm.game_mode_type in self.gym_env.PLAY_STATES:
            self.log("***** Out of setplay")
            self.was_set_play_before = False
        
        if not request.world_model.self.is_kickable:
            # if the ball is not kickable, return Intercept
            self.add_intercept_action(actions, wm)
            return actions
        # if the ball is kickable, send observation to the gym env
        action = self.send_state_get_action(request)
        if not isinstance(action,np.ndarray) and  action == -1:
            self.log("***** GOT RESET")
            # is from reset
            return actions
        # if action[2] == 1:
        #     self.log("***** GOT SHOOT")
        #     actions.actions.append(pb2.PlayerAction(helios_shoot=pb2.HeliosShoot()))
        #     return actions
        self.log(f"***** GOT ACTION: {action}")
        selected_kick_action: pb2.PlayerAction = self.gym_env.gym_action_to_soccer_action(action, request.world_model)
        # convert the action to the grpc action
        actions.actions.append(selected_kick_action)
        return actions

    def add_intercept_action(self, actions, wm):
        self.log("***** BALL NOT KICKABLE, INTERCEPT")
        intercept_action = pb2.Body_Intercept(save_recovery=False, face_point=wm.ball.position)
        actions.actions.append(pb2.PlayerAction(body_intercept=intercept_action))

    def send_state_get_action(self, request):
        self.gym_env.clear_actions_queue()
        self.gym_env.clear_observation_queue()
        self.gym_env.observation_queue.put(request,block=True)
        action = self.gym_env.player_action_queue.get(block = True)
        return action

def serve(gym_env:ContinuousPenaltyEnv,trainer_started:threading.Event):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=22))
    pb2_grpc.add_GameServicer_to_server(GymGame(gym_env,trainer_started), server)
    server.add_insecure_port('localhost:50051')
    server.start()
    print("Decision Server started. Listening on port 50051...")
    try:
        while True:
            time.sleep(60 * 60 * 24)  # Sleep for a day or any desired interval
    except KeyboardInterrupt:
        print("Shutting down the server...")
        server.stop(0)

if __name__ == "__main__":
    # gym_env = DribbleAndShootAngleDiscretizationEnv(verbose=DEBUG_GYM)
    gym_env = ContinuousPenaltyEnv(verbose=DEBUG_GYM)
    trainer_started = threading.Event()
    server_thread = threading.Thread(target=serve, args=(gym_env,trainer_started))
    server_thread.start()
    print("Await trainer")
    
    trainer_started.wait()
    # gym_env.reset()
    checkpoint_callback = CheckpointCallback(5_000,"intermediate_models/DDPG_YuShan_Learn","DDPG",False,False,2)
    
    # log_callback = DDPGCallback()
    # callback_list = CallbackList([checkpoint_callback,log_callback])
    # logger = configure("logs/DQN_discretization_tensorboard/",["stdout","tensorboard","csv"])
    logger = configure("logs/DDPG_YuShanLearn_tesnorboard/",["stdout","tensorboard","csv"])
    model = DDPG('MlpPolicy', gym_env,tensorboard_log="./logs/DDPG_YuShanLearn_tesnorboard/")
    model.set_logger(logger)
    model = model.learn(2_000_000, progress_bar=True,callback=checkpoint_callback)
    model.save("final_models/DDPG_YuShan_Learn")
    print("Model trained")
    print("?????????????????????????????????????????????????????????????????????????")
    observation, _ = gym_env.reset()
    while server_thread.is_alive():
        # get action from the model
        action, _ = model.predict(observation)
        # action = gym_env.action_space.sample()
        print(f"Action: {action}")
        # get observation from the environment 
        observation, reward, terminated ,truncated, info = gym_env.step(action)
        print(f"Observation: {observation}, Reward: {reward}, terminated: {terminated}, Info: {info}")
        if terminated or truncated:
            observation, info = gym_env.reset()
            print("Environment reset")


