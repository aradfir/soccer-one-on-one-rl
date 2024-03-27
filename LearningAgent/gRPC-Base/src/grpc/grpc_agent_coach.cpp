#include "grpc_agent_coach.h"
// #include "state_generator.h"

#include <rcsc/player/say_message_builder.h>
#include <rcsc/common/player_param.h>
#include "coach/sample_coach.h"
#include <chrono>
#include <rcsc/common/logger.h>
#include "grpc/state_generator.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#define DEBUG

#ifdef DEBUG
#define LOG(x) std::cout << x << std::endl
#define LOGV(x) std::cout << #x << ": " << x << std::endl
#else
#define LOG(x)
#define LOGV(x)
#endif


GrpcAgentCoach::GrpcAgentCoach() {
    agent_type = protos::AgentType::CoachT;
}

void GrpcAgentCoach::init(rcsc::CoachAgent * agent,
                            std::string target,
                            int port,
                            bool use_same_grpc_port,
                            bool add_20_to_grpc_port_if_right_side){
    M_agent = agent;
    if (add_20_to_grpc_port_if_right_side)
        if (M_agent->world().ourSide() == rcsc::SideID::RIGHT)
            port += 20;

    if (!use_same_grpc_port){
        port += 13;
    }

    target += ":" + std::to_string(port);

    channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    stub_ = Game::NewStub(channel);
}

void GrpcAgentCoach::getActions() const{
    LOG("getAction Started");
    auto agent = M_agent;
    LOGV(agent->world().time().cycle());
    State state = generateState();
    state.set_agent_type(protos::AgentType::CoachT);
    protos::CoachActions actions;
    ClientContext context;
    Status status = stub_->GetCoachActions(&context, state, &actions);
    if (!status.ok()) {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        return;
    }

    for(int i = 0; i < actions.actions_size(); i++){
        auto action = actions.actions(i);
        switch (action.action_case())
        {
        case CoachAction::kChangePlayerTypes:
        {
            const auto & changePlayerTypes = action.change_player_types();
            const auto & playerType = changePlayerTypes.type();
            const auto & unum = changePlayerTypes.uniform_number();

            agent->doChangePlayerType(unum, playerType);
            break;
        }
        case CoachAction::kDoHeliosSayPlayerTypes:
        {
            auto sample_coach = dynamic_cast<SampleCoach *>(agent);
            sample_coach->sayPlayerTypes();
            break;
        }
        case CoachAction::kDoHeliosSubstitute:
        {
            auto sample_coach = dynamic_cast<SampleCoach *>(agent);
            sample_coach->doSubstitute();
            break;
        }
        default:
        {
            LOG("unknown action");
            break;
        }
        }
    } 
}

State GrpcAgentCoach::generateState() const {
    auto & wm = M_agent->world();
    protos::WorldModel * worldModel = StateGenerator::convertCoachWorldModel(wm);
    State state;
    state.set_allocated_world_model(worldModel);
    return state;
}