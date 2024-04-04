from gym_envs.discrete_env import DiscretePenaltyEnv
import service_pb2 as pb2
import gymnasium as gym
from pyrusgeom.vector_2d import Vector2D as V2D

class DribbleAndShootAngleDiscretizationEnv(DiscretePenaltyEnv):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.dist_to_goalie_factor_mult = 0
        self.SHOOT_POINTS = 6
        self.POSSIBLE_KICK_VELS = [.15,.25,.5]
        self.FRONT_ANGLE_DIVS = 9
        self.FRONT_ANGLE_STEPS = 180/(self.FRONT_ANGLE_DIVS-1)
        self.POSSIBLE_DRIBBLE_ANGLES = [-90+i*self.FRONT_ANGLE_STEPS for i in range(self.FRONT_ANGLE_DIVS)] + [-135,135]

        self.action_space = gym.spaces.Discrete(len(self.POSSIBLE_DRIBBLE_ANGLES) * len(self.POSSIBLE_KICK_VELS) + self.SHOOT_POINTS)
    
    def get_shoot_action(self, action, wm:pb2.WorldModel):
        epsilon = 0.05
        kick_speed = self.server_param.ball_speed_max 
        goal_x = self.server_param.pitch_half_length
        post_y = self.server_param.goal_width/2
        # divide goal into SHOOT_POINTS divs, pick the selected one
        top_shoot_reference = -post_y + epsilon
        actual_shoot_range = self.server_param.goal_width - epsilon*2
        step =  actual_shoot_range / (self.SHOOT_POINTS - 1)
        selected_y = top_shoot_reference + action * step
        # print(f"Action {action}, Y = {selected_y}, ")
        shoot_target = pb2.Vector2D(x=goal_x,y= selected_y)
        kick = pb2.Body_SmartKick(target_point=shoot_target,first_speed=kick_speed,max_steps=1,first_speed_threshold=0.5)
        return pb2.PlayerAction(body_smart_kick=kick)
    
    def get_dribble_action(self, action, wm:pb2.WorldModel):
        angle = action//len(self.POSSIBLE_KICK_VELS)
        kick = action % len(self.POSSIBLE_KICK_VELS)
        kick_vel = self.POSSIBLE_KICK_VELS[kick] * self.server_param.ball_speed_max
        final_angle = self.POSSIBLE_DRIBBLE_ANGLES[angle]
        pos = wm.self.position
        target = V2D.polar2vector(10,final_angle)+ V2D(x=pos.x,y=pos.y)
        kick = pb2.Body_KickOneStep(first_speed=kick_vel,target_point=pb2.Vector2D(x=target.x(),y=target.y()),force_mode=True)
        action = pb2.PlayerAction(body_kick_one_step=kick)
        return action

    
    def gym_action_to_soccer_action(self, action, wm: pb2.WorldModel):
        if not action < self.SHOOT_POINTS:
            return self.get_dribble_action(action - self.SHOOT_POINTS, wm)
        return self.get_shoot_action(action, wm)
        