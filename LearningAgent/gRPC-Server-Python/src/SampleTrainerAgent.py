from abc import ABC
from src.IAgent import IAgent
import service_pb2 as pb2


class SampleTrainerAgent(IAgent, ABC):
    def __init__(self):
        super().__init__()
        self.serverParams: pb2.ServerParam = None
        self.playerParams: pb2.PlayerParam = None
        self.playerTypes: dict[pb2.PlayerType] = {}
        self.wm: pb2.WorldModel = None
        self.first_substitution = True
    
    def get_actions(self, wm:pb2.WorldModel) -> pb2.TrainerActions:
        self.wm = wm
        
        actions = pb2.TrainerActions()
        print(f'cycle: {self.wm.cycle}')
        print(f'cycle: {self.wm.ball.position.x}, {self.wm.ball.position.y}')
        
        if self.wm.cycle % 100 == 0:
            print("Sending trainer action")
            actions.actions.append(
                pb2.TrainerAction(
                    do_move_ball=pb2.DoMoveBall(
                        position=pb2.Vector2D(
                            x=0,
                            y=0
                        ),
                        velocity=pb2.Vector2D(
                            x=0,
                            y=0
                        ),
                    )
                )
            )
        return actions
    
    def set_params(self, params):
        if isinstance(params, pb2.ServerParam):
            self.serverParams = params
        elif isinstance(params, pb2.PlayerParam):
            self.playerParams = params
        elif isinstance(params, pb2.PlayerType):
            self.playerTypes[params.id] = params
        else:
            raise Exception("Unknown params type")