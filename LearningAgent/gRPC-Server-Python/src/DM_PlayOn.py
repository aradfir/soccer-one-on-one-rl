import service_pb2 as pb2
from src.IDecisionMaker import IDecisionMaker
from src.IAgent import IAgent
from src.DM_WithBall import WithBallDecisionMaker
from src.DM_NoBall import NoBallDecisionMaker


class PlayOnDecisionMaker(IDecisionMaker):
    def __init__(self):
        self.withBallDecisionMaker = WithBallDecisionMaker()
        self.noBallDecisionMaker = NoBallDecisionMaker()
        pass
    
    def make_decision(self, agent: IAgent):
        # agent.addAction(pb2.PlayerAction(dash=pb2.Dash(power=100, relative_direction=30)))
        if agent.wm.self.is_kickable:
            self.withBallDecisionMaker.make_decision(agent)
            # agent.add_action(pb2.PlayerAction(helios_chain_action=pb2.HeliosChainAction(
            #     cross=True,
            #     lead_pass=True,
            #     direct_pass=True,
            #     through_pass=True,
            #     short_dribble=True,
            #     long_dribble=True,
            # )))
        else:
            self.noBallDecisionMaker.make_decision(agent)