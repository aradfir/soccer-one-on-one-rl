from gym_envs.discrete_env import DiscretePenaltyEnv
import service_pb2 as pb2
import gymnasium as gym

class DribbleAndShootEnv(DiscretePenaltyEnv):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.dist_to_goalie_factor_mult = 0
        self.SHOOT_POINTS = 5
        self.POSSIBLE_KICK_VELS = [.25,.45]
        self.ANGLE_DIVS=12
        self.action_space = gym.spaces.Discrete(self.ANGLE_DIVS * len(self.POSSIBLE_KICK_VELS) + self.SHOOT_POINTS)
    
    def gym_action_to_soccer_action(self, action, wm: pb2.WorldModel):
        if not action < self.SHOOT_POINTS:
            return super().gym_action_to_soccer_action(action - self.SHOOT_POINTS, wm)
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
        