import gymnasium as gym
from queue import Queue
import numpy as np
from service_pb2 import GameModeType, ServerParam, PlayerType, PlayerParam, Side, State, TrainerAction, TrainerActions, Vector2D, WorldModel
import service_pb2 as pb2
from service_pb2 import Body_KickOneStep
from pyrusgeom.vector_2d import Vector2D as V2D
class CustomGymEnv(gym.Env):
    def __init__(self, verbose = False) -> None:
        super().__init__()
        ### CONFIGURATION ###
        self.ANGLE_DIVS = 12
        # self.RELATIVE_KICK_ANGLES= [0,15,-15,30,-30,60,-60,90,-90,135,-135,180]
        self.POSSIBLE_KICK_VELS = [.1,.2,.3,.4,.5,.6,.7,.8] # relative to ball speed max
        self.OOB_REWARD = -300
        self.GOAL_REWARD = 1500
        self.GOALIE_CATCH_REWARD = -200
        self.TIMEOUT_REWARD = -150
        self.STEP_REWARD = -10
        self.TIMEUT_CYCLES = 150 # as set for penalty mode
        self.TERMINAL_STATES = [GameModeType.AfterGoal_, GameModeType.FreeKick_, GameModeType.CornerKick_, GameModeType.GoalieCatch_, GameModeType.KickIn_,GameModeType.KickOff_, GameModeType.GoalKick_]
        self.PLAY_STATES = [GameModeType.PlayOn]
        #######################
        self.player_action_queue = Queue(1)
        self.trainer_action_queue = Queue(10)
        self.observation_queue = Queue(1)
        self.episode_reward = 0
        self.old_observation = None
        self.server_param: ServerParam = None
        self.player_param: PlayerParam = None
        self.player_type: PlayerType = None
        self.ANGLE_STEP = 360/self.ANGLE_DIVS
        self.verbose = verbose
        self.episode_start_cycle = 999
        
        # observation: normalized self pos (polar to goal) (r, theta), normalized ball pos(x,y), normalized goalie pos(x,y), goalie body relative to ball (theta), goalie pos relative to player(r, theta)
        self.observation_space = gym.spaces.Box(low=np.array([0,-1,-1,-1,-1,-1,-1,0,-1]),high=np.array([1,1,1,1,1,1,1,1,1])
                                                ,shape=(9,),dtype=np.float64)
        # action space: kick, 18 angles, 5 power levels
        # use multidescrete
        # self.action_space = gym.spaces.MultiDiscrete([len(self.RELATIVE_KICK_ANGLES), len(self.POSSIBLE_KICK_VELS)])
        # self.action_space = gym.spaces.MultiDiscrete([self.ANGLE_DIVS,len(self.POSSIBLE_KICK_VELS)])
        self.action_space = gym.spaces.Box(low=np.array([-1,.15]),high=np.array([1,1]),shape=(2,),dtype=np.float64)


    def is_timed_out(self, current_cycle):
        return current_cycle - self.episode_start_cycle >= self.TIMEUT_CYCLES
    
    
    def get_their_goalie(self, observation: State) -> pb2.Player:
        wm = observation.world_model
        goalie_unum = wm.their_goalie_uniform_number
        if goalie_unum <= 0 or goalie_unum > 11:
            if self.verbose:
                print("CANT SEE OPP GOALIE UNUM")
            return None
        
        goalie = wm.their_players_dict[goalie_unum]
        if goalie.uniform_number != goalie_unum:
            if self.verbose:
                print("OPP GOALIE DOESNT MATCH")
            return None
        return goalie
    
    def get_goalie_pos(self, observation: State) -> Vector2D:
        goalie = self.get_their_goalie(observation)
        if not goalie:
            return Vector2D(x=-999., y= -999.)
        return goalie.position



    def observation_to_ndarray(self, observation: State) -> np.ndarray:
        wm = observation.world_model
        self_pos = wm.self.position
        ball_pos = wm.ball.position
        opp_pos = self.get_goalie_pos(observation)
        
        hl = self.server_param.pitch_half_length
        hw = self.server_param.pitch_half_width
        self_angle = (V2D(x=52.5,y=0) - V2D(x=self_pos.x,y=self_pos.y)).th()
        normalized_self_pos = [self.dist_to_goal(self_pos)/120, self_angle.degree_()/90]
        normalized_ball_pos = [ball_pos.x/hl, ball_pos.y/hw]
        
        if abs(opp_pos.x) <= hl:
            opp = self.get_their_goalie(observation)
            normalized_opp_pos = [opp_pos.x/hl, opp_pos.y/hw]
            relative_angle = [(opp.angle_from_ball - opp.body_direction)/180]
            goalie_dist = opp.dist_from_self/100
            goalie_angle = opp.angle_from_self/180
            relative_goalie_pos = [goalie_dist,goalie_angle]
        else:
            # cant see opp
            normalized_opp_pos = [1, 0]
            relative_angle = [0]


        if self.verbose:
            print(f"Self Pos: {normalized_self_pos}, Ball Pos: {normalized_ball_pos}, Opp Pos: {normalized_opp_pos}, Relative Opp angle: { relative_angle}")
        return np.array(normalized_self_pos + normalized_ball_pos + normalized_opp_pos+relative_angle+relative_goalie_pos)
    
    def calculate_reward(self, old_observation:State, action:np.ndarray, observation:State) -> float:
        wm = observation.world_model
        gamemode = wm.game_mode_type
        # todo: hack to find which goal was scored in since goal_l or goal_r dont send observations.
        if gamemode == GameModeType.KickOff_:
            last_ball_x = old_observation.world_model.ball.position.x
            if last_ball_x > 0:
                actual_ball = Vector2D(x=53.5,y=0)
            else:
                actual_ball = Vector2D(x=-53.5,y=0)
            ball_pos = actual_ball
        else:
            ball_pos = wm.ball.position

        hl = self.server_param.pitch_half_length
        hw = self.server_param.pitch_half_width
        if gamemode == GameModeType.KickIn_ or gamemode == GameModeType.CornerKick_ or gamemode == GameModeType.GoalKick_ or abs(ball_pos.y) >= hw:
            return self.OOB_REWARD
        if gamemode == GameModeType.FreeKick_ or gamemode == GameModeType.GoalieCatch_:
            return self.GOALIE_CATCH_REWARD
        if ball_pos.x >= hl:
            if abs(ball_pos.y) < self.server_param.goal_width/2:
                # ITS A GOAL :D
                return self.GOAL_REWARD
            else:
                # out of bounds (side of goal)
                return self.OOB_REWARD
    
        
        if self.is_timed_out(wm.cycle):
            return self.TIMEOUT_REWARD
        
        goalie = self.get_their_goalie(observation)
        
        old_goalie = self.get_their_goalie(old_observation)
        old_ball_pos = old_observation.world_model.ball.position
        # close to goal is good
        dist_to_goal_factor = -self.dist_to_goal(ball_pos) + self.dist_to_goal(old_ball_pos)
        # close to goalie is risky
        dist_to_goalie_factor = goalie.dist_from_ball - old_goalie.dist_from_ball
        
        previous_cycle_shoot = 0
        # if action[2] == 1:
        #     # we shot before but didnt do anything
        #     previous_cycle_shoot = -5
        
        # print(f'Ball Pos:({ball_pos.x},{ball_pos.y}), old ball pos : ({old_ball_pos.x},{old_ball_pos.y})')
        return self.STEP_REWARD + 2.5*dist_to_goal_factor + 0.5* dist_to_goalie_factor + previous_cycle_shoot
    

    def dist_to_goal_square(self, pos:Vector2D):
        hl = self.server_param.pitch_half_length
        return (hl- pos.x) **2 + pos.y**2
    
    def dist_to_goal(self, pos:Vector2D):
        return np.sqrt(self.dist_to_goal_square(pos))

    def wait_for_observation_and_return(self):
        observation = self.observation_queue.get(block=True)
        self.old_observation = observation
        return self.observation_to_ndarray(observation), {}
    
    def clear_actions_queue(self):
        with self.player_action_queue.mutex:
            self.player_action_queue.queue.clear()

    def clear_observation_queue(self):
        with self.observation_queue.mutex:
            self.observation_queue.queue.clear()

    def do_action(self, action, clear_actions: bool = False):
        if clear_actions:
            self.clear_actions_queue()
        self.player_action_queue.put(action,block=False)

    def gym_action_to_soccer_action(self, action, wm:WorldModel):
        # angle = -np.pi + action[0] * self.ANGLE_STEP
        # angle = -180 + action[0] * self.ANGLE_STEP
        # power = (self.POSSIBLE_KICK_VELS[action[1]]) * self.server_param.ball_speed_max
        # print(f"Body Dir: {wm.self.body_direction}")
        # pos = wm.self.position
        # target = V2D.polar2vector(10,absolute_angle)+ V2D(x=pos.x,y=pos.y)
        
        # print(f"Absolute Angle: {absolute_angle}, Power: {power}")
        
        # return Body_KickOneStep(first_speed=power,target_point=Vector2D(x=target.x(),y=target.y()),force_mode=True)
        angle = action[0] * 180
        power = action[1] * self.server_param.ball_speed_max
        
        # angle = self.RELATIVE_KICK_ANGLES[action[0]]
        pos = wm.self.position
        target = V2D.polar2vector(10,angle)+ V2D(x=pos.x,y=pos.y)
        return Body_KickOneStep(first_speed=power,target_point=Vector2D(x=target.x(),y=target.y()),force_mode=True)



    def step(self, action):
        self.clear_observation_queue()
        self.do_action(action)
        observation:State = self.observation_queue.get()
        wm = observation.world_model
        game_mode = wm.game_mode_type
        reward = 0
        if self.old_observation is not None:
            reward = self.calculate_reward(self.old_observation, action, observation)
        elif self.verbose:
            print("NOT CALCULATING REWARD AFTER RESET")
        self.episode_reward += reward
        
        terminated = game_mode in self.TERMINAL_STATES
        truncated = self.is_timed_out(wm.cycle)
        self.print_debug(wm, reward, terminated, truncated)
        self.old_observation = observation
        return self.observation_to_ndarray(observation), reward, terminated, truncated, {}

    def print_debug(self, wm, reward, terminated, truncated):
        if self.verbose:
            print(f"cycle:{wm.cycle}, reward = {reward}")
            if terminated:
                print("DONE")
            if truncated:
                print("TIMEOUT")
    
    def get_trainer_reset_commands(self) -> TrainerActions:
        actions = TrainerActions()
        zero_vec = Vector2D(x=0.,y=0.)
        pen_point = Vector2D(x =-1 *( self.server_param.pitch_half_length/2 - self.server_param.pen_dist_x), y=0)
        goal_vec = Vector2D(x=50,y=0)
        player_vec = Vector2D(x=pen_point.x - 2,y=0.)
        actions.actions.append(TrainerAction(do_change_mode=pb2.DoChangeMode(game_mode_type=GameModeType.PlayOn,side=Side.LEFT)))
        actions.actions.append(TrainerAction(do_move_ball=pb2.DoMoveBall(position=pen_point,velocity=zero_vec)))
        actions.actions.append(TrainerAction(do_recover=pb2.DoRecover()))
        actions.actions.append(TrainerAction(do_move_player=pb2.DoMovePlayer(our_side=True, uniform_number= 1, position= player_vec, body_direction=0)))
        actions.actions.append(TrainerAction(do_move_player=pb2.DoMovePlayer(our_side=False, uniform_number= 1, position= goal_vec, body_direction=0)))
        return actions

    
    def reset(self, seed = -1):
        print(f"Episode reward:{self.episode_reward}")
        if self.verbose:
            print("================================")
        self.episode_reward = 0
        self.old_observation = None
        self.clear_actions_queue()
        self.clear_observation_queue()
        if self.trainer_action_queue.empty:
            self.trainer_action_queue.put(seed)
        # reset action sent, to unblock player action
        self.do_action([-1,-1], clear_actions=True)
        return self.wait_for_observation_and_return()