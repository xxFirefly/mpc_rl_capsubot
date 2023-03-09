from casadi_model import make_casadi_model_integrator, calculate_control_with_single_shooting, calculate_control_with_multiple_shooting, test_integrator
import os
import sys
from collections import deque
from datetime import datetime
import pickle

import casadi as ca

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPT_FOLDER, ".."))
from capsubot_env.capsubot import Capsubot

# Main problem parameters
INIT_STATE = [0.0, 0.0, 0.0, 0.0] # x x_dot xi xi_dot
TERMINAL_TIME = 20.0 # seconds
MPC_PREDICTION_HORIZON = 0.3 # seconds
MPC_PREDICTION_HORIZON_PART_TO_APPLY = 1.0 # from 0.0 to 1.0

# Solver parameters
CAPSUBOT_MODEL_TYPE = 0
FIXED_CONTROL_DURATION = 0.005
MAX_INTEGRATION_DT = FIXED_CONTROL_DURATION # could be lower

IPOPT_OPTS = {
    "ipopt": {
        "max_iter": 50,
        "print_level": 1,
        "acceptable_tol": 1e-8,
        "acceptable_obj_change_tol": 1e-4,
    },
}

if __name__ == "__main__":
    model = Capsubot(frame_skip=1, model=CAPSUBOT_MODEL_TYPE, dt=MAX_INTEGRATION_DT)
    model_integrator = make_casadi_model_integrator(
        model=model, integration_duration=FIXED_CONTROL_DURATION, max_integration_dt=MAX_INTEGRATION_DT
    )
    test_integrator(F=model_integrator, model=model, integration_duration=FIXED_CONTROL_DURATION)

    X = ca.DM(INIT_STATE)
    current_time = 0.0
    objective = 0.0

    poses = deque()
    controls = deque()
    times = deque()
    try:
        while current_time < TERMINAL_TIME:
            sol = calculate_control_with_single_shooting(init_state=X, init_t=current_time, init_objective=objective, optimization_duration=MPC_PREDICTION_HORIZON, integration_duration=FIXED_CONTROL_DURATION, model=model, F=model_integrator, ipopt_opts=IPOPT_OPTS)

            n_steps = int(MPC_PREDICTION_HORIZON_PART_TO_APPLY*sol['x'].shape[0])
            for k in range(n_steps):
                Ik = model_integrator(x0=X, p=sol['x'][k])
                X = Ik['xf']
                objective += Ik['qf']
                current_time += FIXED_CONTROL_DURATION
                poses.append(X.full().squeeze())
                controls.append(float(sol['x'][k]))
                times.append(current_time)
            print(current_time, X, float(X[0])/current_time)
    except Exception as e:
        print(e)

    data = {"states": poses, "actions": controls, "timestamps": times}
    time_now = datetime.now()
    filename = f"max_average_sh_{CAPSUBOT_MODEL_TYPE}_dur_{int(MPC_PREDICTION_HORIZON*1000)}_T_{int(TERMINAL_TIME*1000)}_{time_now.strftime('dym_%y_%m_%d_hm_%H_%M')}.pkl"
    git_dir = os.path.join(SCRIPT_FOLDER, "..", "..", "results")
    resulted_path = os.path.join(git_dir, filename)
    with open(resulted_path, "wb") as fp:
        pickle.dump(data, fp)
    print(f"results saved to {resulted_path}")