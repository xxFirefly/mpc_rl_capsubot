from capsubot_env.capsubot import Capsubot
from typing import List
from brute_mpc import *
import pickle

force_encodings = make_force_encodings(10)
fixed_T = 0.10; model_type = 0
frame_skip = int(0.1/ MIN_DT)
mpc_duration = fixed_T  # 5*fixed_T
print(frame_skip)
agent = Capsubot(dt=MIN_DT, frame_skip=frame_skip, model=model_type, log=True)
data = {"states": [], "actions": [], "timestamps": []}
idx = 0
while agent.get_total_time < 20.0:
    force = calculate_mpc_force_max_average(
        current_state=agent.get_state,
        duration=mpc_duration,
        model_type=model_type,
        T=fixed_T,
        total_time=agent.get_total_time,
        force_encodings=force_encodings,
    )
    agent.step_force(unit_force=force)

    data["states"] = agent.x_buffer
    data["actions"] = agent.action_buffer
    data["timestamps"] = agent.total_time_buffer
    print(agent.get_total_time,  agent.x_buffer[-1][0], agent.x_buffer[-1][0]/agent.get_total_time)
    print(f"mpc_2max_average_{model_type}_fs_{frame_skip}_dur_{int(mpc_duration*1000)}_T_{int(fixed_T*1000)}.pkl")
    with open(
        f"mpc_2max_average_{model_type}_fs_{frame_skip}_dur_{int(mpc_duration*1000)}_T_{int(fixed_T*1000)}.pkl", "wb"
    ) as fp:
        pickle.dump(data, fp)
