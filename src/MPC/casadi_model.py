import os
import sys
from typing import Callable, List, Dict

import casadi as ca

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPT_FOLDER, ".."))
from capsubot_env.capsubot import Capsubot


def make_casadi_model_integrator(model: Capsubot, integration_duration: float, max_integration_dt: float) -> Callable:
    x = ca.SX.sym("x")
    x_dot = ca.SX.sym("x_dot")
    xi = ca.SX.sym("xi")
    xi_dot = ca.SX.sym("xi_dot")
    state = ca.vertcat(x, x_dot, xi, xi_dot)
    u = ca.SX.sym("u")

    friction = -model._N * 2 / ca.pi * ca.atan(x_dot * 10e5)

    dx = x_dot
    dx_dot = (model._stiffness * xi - u + friction) / model._M
    dxi = xi_dot
    dxi_dot = (-model._stiffness * xi + u) / model._m - dx_dot

    # Resulted system of ode equations
    ode = ca.vertcat(dx, dx_dot, dxi, dxi_dot)

    # Objective function
    L = -x_dot

    dae = {"x": state, "p": u, "ode": ode, "quad": L}
    F = ca.integrator("F", "cvodes", dae, 0, integration_duration, {"max_step_size": max_integration_dt})
    return F


def test_integrator(F: Callable, model: Capsubot, integration_duration: float):
    allowed_error = 0.005
    if model.model_type == 1:
        T = 0.0441
        tau = 0.785
        target_av_v = -0.0748
    elif model.model_type == 0:
        T = 0.097
        tau = 0.14
        target_av_v = 0.01736

    X = ca.DM.zeros(4)
    Q = ca.DM.zeros(1)
    N_steps = int(2.0 / integration_duration)
    time = 0.0

    for i in range(N_steps):
        time = i * integration_duration
        force = model._force_max if time % T / T < tau else 0.0
        Fk = F(x0=X, p=force)
        X = Fk["xf"]
        Q += Fk["qf"]

    av_v = float(X[0]) / time
    error = abs(av_v - target_av_v)
    assert (
        error < allowed_error
    ), f"error: {error:.3f} allowed_error: {allowed_error:.3f} Q:{float(Q):.3f} vel:{av_v:.3f}"
    print(f"canonical velocity {target_av_v:.3f}, casadi model velocity: {av_v:.3f}")


def make_initial_guess(integration_duration: float, model: Capsubot, number_of_steps: int, init_t: float) -> ca.DM:
    init_guess = ca.DM.zeros(number_of_steps)

    if model.model_type == 0:
        T = 0.14
        tau = 0.2
    elif model.model_type == 1:
        T = 0.0441
        tau = 0.785
    else:
        return init_guess
    for k in range(number_of_steps):
        time = init_t + k * integration_duration
        part_period = time % T / T
        init_guess[k] = model._force_max if part_period < tau else 0.0

    return init_guess


# Probably need to redo this
def calculate_initial_objective(
    integrator: Callable, init_x: List, init_guess: List, number_of_steps: int, init_Q: List
) -> float:
    X = init_x
    Q = init_Q
    for k in range(number_of_steps):
        Ik = integrator(x0=X, p=init_guess[k])
        X = Ik["xf"]
        Q += -Ik["qf"]
    return Q


def calculate_control_with_single_shooting(
    F: Callable,
    init_state: List,
    init_objective: float,
    init_t: float,
    optimization_duration: float,
    integration_duration: float,
    model: Capsubot,
    ipopt_opts: Dict,
) -> Dict:
    U_N = int(optimization_duration / integration_duration)
    U = ca.MX.sym("U", U_N)

    # Derive nlp problem
    J = init_objective
    X = init_state
    g = []
    for k in range(U_N):
        Ik = F(x0=X, p=U[k])
        X = Ik["xf"]
        J += 1000*Ik["qf"]

    g = ca.vertcat(*g)
    nlp = {"x": U, "f": J, 'g' : g}
    solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

    init_guess = make_initial_guess(
        integration_duration=integration_duration, model=model, number_of_steps=U_N, init_t=init_t
    )
    solution = solver(lbx=0.0, ubx=model._force_max, lbg=0.0, ubg=0.0, x0=init_guess)
    return solution

def calculate_control_with_multiple_shooting(
    F: Callable,
    init_state: List,
    init_objective: float,
    init_t: float,
    optimization_duration: float,
    integration_duration: float,
    model: Capsubot,
    ipopt_opts: Dict,
) -> Dict:
    U_N = int(optimization_duration / integration_duration)
    state_size = init_state.shape[0]

    U0 = make_initial_guess(
        integration_duration=integration_duration, model=model, number_of_steps=U_N, init_t=init_t
    )

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = init_objective
    g=[]
    lbg = []
    ubg = []

    Xk = ca.MX.sym('X0', state_size)
    w += [Xk]
    lbw += list(init_state.T.full().squeeze())
    ubw += list(init_state.T.full().squeeze())
    w0 += list(init_state.T.full().squeeze())
    for k in range(U_N):
        Uk = ca.MX.sym("U_" + str(k))
        # Add control varibale to optimization
        w += [Uk]
        lbw += [0.0]
        ubw += [model._force_max]
        w0 += list(U0.full()[k])

        # Calculate resulted state for boundary condition
        Ik = F(x0=Xk, p=Uk)
        X_u_end = Ik['xf']
        J += 1000*Ik["qf"]

        Xk = ca.MX.sym("X_" + str(k + 1), state_size)
        w += [Xk]
        lbw += [-ca.inf] * state_size
        ubw += [ca.inf] * state_size
        w0 += [0.0] * state_size
        g += [X_u_end - Xk]
        lbg += [0.0] * state_size
        ubg += [0.0] * state_size

    nlp = {"x": ca.vertcat(*w), "f": J, 'g' : ca.vertcat(*g)}
    solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    # Leave only control variables solution
    solution['x'] = solution['x'][state_size::(state_size + 1)]
    return solution

def calculate_control_with_single_shooting_problem_2(
    F: Callable,
    init_state: List,
    target_state: List,
    init_objective: float,
    init_t: float,
    optimization_duration: float,
    integration_duration: float,
    model: Capsubot,
    ipopt_opts: Dict,
) -> Dict:
    U_N = int(optimization_duration / integration_duration)
    U = ca.MX.sym("U", U_N)

    # Weights for objective function
    P = ca.MX.eye(4)
    P[0] = 1000
    target_state = ca.DM(target_state)


    # Derive nlp problem
    J = init_objective
    X = init_state

    for k in range(U_N):
        Ik = F(x0=X, p=U[k])
        X = Ik["xf"]
        # J += 1000*U[k]*(model._force_max - U[k])

    state_error = target_state - X
    J += state_error.T@P@state_error

    nlp = {"x": U, "f": J}
    solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

    init_guess = make_initial_guess(
        integration_duration=integration_duration, model=model, number_of_steps=U_N, init_t=init_t
    )
    solution = solver(lbx=0.0, ubx=model._force_max, x0=init_guess)
    return solution

def calculate_control_with_multiple_shooting_problem_2(
    F: Callable,
    init_state: List,
    target_state: List,
    init_objective: float,
    init_t: float,
    optimization_duration: float,
    integration_duration: float,
    model: Capsubot,
    ipopt_opts: Dict,
) -> Dict:
    U_N = int(optimization_duration / integration_duration)
    U0 = make_initial_guess(
        integration_duration=integration_duration, model=model, number_of_steps=U_N, init_t=init_t
    )
    state_size = init_state.shape[0]

    # Weights for objective function
    P = ca.MX.eye(4)
    P[0] = 1000
    target_state = ca.DM(target_state)

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = init_objective
    g=[]
    lbg = []
    ubg = []

    Xk = ca.MX.sym('X0', state_size)
    w += [Xk]
    lbw += list(init_state.T.full().squeeze())
    ubw += list(init_state.T.full().squeeze())
    w0 += list(init_state.T.full().squeeze())
    for k in range(U_N):
        Uk = ca.MX.sym("U_" + str(k))
        # Add control varibale to optimization
        w += [Uk]
        lbw += [0.0]
        ubw += [model._force_max]
        w0 += list(U0.full()[k])

        # Calculate resulted state for boundary condition
        Ik = F(x0=Xk, p=Uk)
        X_u_end = Ik['xf']
        state_error = target_state - X_u_end
        J = state_error.T@P@state_error

        Xk = ca.MX.sym("X_" + str(k + 1), state_size)
        w += [Xk]
        lbw += [-ca.inf] * state_size
        ubw += [ca.inf] * state_size
        w0 += [0.0] * state_size
        g += [X_u_end - Xk]
        lbg += [0.0] * state_size
        ubg += [0.0] * state_size

    nlp = {"x": ca.vertcat(*w), "f": J, 'g' : ca.vertcat(*g)}
    solver = ca.nlpsol("solver", "ipopt", nlp, ipopt_opts)

    solution = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    # Leave only control variables solution
    solution['x'] = solution['x'][state_size::(state_size + 1)]
    return solution

if __name__ == "__main__":
    model_type = 0
    integration_duration = 0.005
    max_integration_dt = 1.0  # # 0.01/200.0
    model = Capsubot(dt=0.01 / 200.0, frame_skip=1, model=model_type)
    F = make_casadi_model_integrator(
        model=model, integration_duration=integration_duration, max_integration_dt=max_integration_dt
    )
    test_integrator(F=F, model=model, integration_duration=integration_duration)

    # need to remove this
    opts = {
        "ipopt": {
            "max_iter": 50,
            "print_level": 1,
            "acceptable_tol": 1e-8,
            "acceptable_obj_change_tol": 1e-4,
        },
    }

    X = ca.DM([0.0, 0.0, 0.0, 0.0])
    sol = calculate_control_with_single_shooting(
        F=F,
        init_state=X,
        init_objective=0.0,
        init_t=0.0,
        optimization_duration=0.3,
        integration_duration=integration_duration,
        model=model,
        ipopt_opts=opts,
    )

    current_time = 0.0
    n_steps = int(sol["x"].shape[0])
    for k in range(n_steps):
        Ik = F(x0=X, p=sol["x"][k])
        X = Ik["xf"]
        current_time += integration_duration
    print(current_time, X, float(X[0]) / current_time)
