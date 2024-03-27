from abc import ABC
from src.IAgent import IAgent
import service_pb2 as pb2


class SampleCoachAgent(IAgent, ABC):
    def __init__(self):
        super().__init__()
        self.serverParams: pb2.ServerParam = None
        self.playerParams: pb2.PlayerParam = None
        self.playerTypes: dict[pb2.PlayerType] = {}
        self.wm: pb2.WorldModel = None
        self.first_substitution = True
    
    def get_actions(self, wm:pb2.WorldModel) -> pb2.CoachActions:
        self.wm = wm
        
        actions = pb2.CoachActions()
        # if (wm.cycle == 0
        #     and self.first_substitution
        #     and self.playerParams is not None
        #     and len(self.playerTypes.keys()) == self.playerParams.player_types):
            
        #     self.first_substitution = False
        #     for i in range(11):
        #         actions.actions.append(
        #             pb2.CoachAction(
        #                 change_player_types=pb2.ChangePlayerType(
        #                 uniform_number=i+1,
        #                 type=i
        #                 )
        #             )
        #         )

        actions.actions.append(
            pb2.CoachAction(
                do_helios_substitute=pb2.DoHeliosSubstitute()
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