import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from capsubot_env.capsubot import Capsubot, classic_force_model
from capsubot_env.capsubot_env import CapsubotEnv
from typing import List, Callable
from collections import deque

MIN_DT = CapsubotEnv().dt


def compute_trajectory(
    duration: int,
    force_func: Callable[[float], bool],
    dt: int = MIN_DT,
    model_type=0,
    initial_state: List[float] = [0.0, 0.0, 0.0, 0.0],
    initial_time: float = 0.0,
) -> List[List[float]]:
    """
    Returns:
        Agent state as [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
    """
    agent = Capsubot(dt=dt, frame_skip=1, model=model_type)
    agent.reset()
    agent.set_state(initial_state)
    agent.set_time(initial_time)

    while agent._total_time <= initial_time + duration:
        action = force_func(agent.get_total_time)
        agent.step(action=action)
    return np.append([agent.get_total_time], agent.get_state)


def compute_classic_trajectory(
    duration: int, T: int, tau: int, dt: int = 0.01 / 200, model_type=0
) -> List[List[float]]:
    """
    Returns:
        Agent state as [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
    """
    force_func = lambda t: classic_force_model(t, T, tau)
    return compute_trajectory(duration=duration, force_func=force_func, dt=dt, model_type=model_type)


def compute_classic_trajectory_dimless(
    n_periods: int, T: int, tau: int, model_type=0, dim_dt=0.01 / 200
) -> List[List[float]]:
    """
    Returns:
        Agent state as [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
    """
    dummy_agent = Capsubot(dt=0.1, frame_skip=1, model=model_type)
    dim_T = T / dummy_agent.omega
    state = compute_classic_trajectory(
        duration=dim_T * n_periods,
        T=dim_T,
        tau=tau,
        dt=dim_dt,
        model_type=model_type,
    )

    dimless_state = [
        state[0] * dummy_agent.omega,
        state[1] / dummy_agent.L,
        state[2] / dummy_agent.L / dummy_agent.omega,
        state[3] / dummy_agent.L,
        state[4] / dummy_agent.L / dummy_agent.omega,
    ]
    return dimless_state


def compute_optimal_force():
    pass


def compute_metric():
    pass


def test_caspubot_model_0():
    kEps = 0.1
    states = compute_classic_trajectory_dimless(n_periods=50.0, T=1.1 * 2 * np.pi, tau=0.1, model_type=0, dim_dt=MIN_DT)
    av_velocity = states[1] / states[0]
    target_av_velocity = 0.047
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps


def test_caspubot_model_1():
    kEps = 0.1
    states = compute_classic_trajectory_dimless(n_periods=50.0, T=7.95, tau=0.215, model_type=1, dim_dt=MIN_DT)
    av_velocity = states[1] / states[0]
    target_av_velocity = 0.1868
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps

    states = compute_classic_trajectory(duration=2.0, T=0.0441, tau=0.785, dt=MIN_DT, model_type=1)
    av_velocity = states[1] / states[0]
    target_av_velocity = -0.0748
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps
    print(f"\n\nMAX PAPER VEL: {abs(av_velocity)}\n\n")


def test(func):
    func()
    print(f"{func.__name__} is OK")


# Generate all possible forces with fixed number of switches
def add_switch(value: int, array: list, max_len: int, arrays: List[List]):
    if value != -1:
        array.append(value)
    if len(array) == max_len:
        # print(array)
        arrays.append(array.copy())
        return
    add_switch(0, array, max_len, arrays)
    array.pop()
    add_switch(1, array, max_len, arrays)
    array.pop()


def make_force_encodings(num_switches: int) -> List[List]:
    force_encodings = []
    temp = []
    add_switch(-1, temp, num_switches, force_encodings)
    force_encodings.reverse()
    return force_encodings


def force_from_encoding(t, T, force_encoding):
    fraction = t % T / T
    idx = int(fraction * len(force_encoding))
    return force_encoding[idx]


def calculate_mpc_action(
    current_state: List[float],
    model_type: int,
    duration: int,
    force_encodings: List[List],
    T: float = 0.1,
    total_time=0.0,
) -> float:
    # sol = None
    # for idx, force_encoding in enumerate(force_encodings):
    #     force = lambda t : force_from_encoding(t, T=T, force_encoding=force_encoding)
    #     state = compute_trajectory(initial_state=current_state, duration=duration, force_func=force, model_type=model_type)

    #     mean_velocity = abs((state[1] - current_state[0])/state[0])

    #     if sol is None:
    #         sol = (idx, mean_velocity)
    #         continue

    #     if mean_velocity > sol[1]:
    #         sol = (idx, mean_velocity)

    # sol = None
    # for tau in np.linspace(0.0, 1.0, 11):
    #     for step in [1, 2]:
    #         if step == 1:
    #             force = lambda t : classic_force_model(t + total_time, T=T, tau=tau)
    #         elif step == 2:
    #             force = lambda t : 1 - classic_force_model(t + total_time, T=T, tau=tau)

    #         state = compute_trajectory(initial_state=current_state, duration=duration, force_func=force, model_type=model_type)

    #         mean_velocity = abs(state[1]/state[0])

    #         if sol is None:
    #             sol = [mean_velocity, force, tau]

    #         if mean_velocity > sol[0]:
    #             sol = [mean_velocity, force, tau]

    # resulted_action = sol[1](t=0)
    # print(f"action: {resulted_action}, tau: {sol[2]}, vel: {sol[0]}")
    force = lambda t: classic_force_model(t, T=T, tau=(1 - 0.785))
    # print(force(0), total_time, T)
    return force


def calculate_mpc_force(
    current_state: List[float],
    model_type: int,
    duration: int,
    T: float = 0.1,
    total_time=0.0,
) -> float:

    sol = None
    for tau in np.linspace(0.0, 0.5, 101):
        for step in [1]:
            if step == 1:
                force = lambda t, pr=False: classic_force_model(t, T=T, tau=tau, pr=pr)
            elif step == 2:
                force = lambda t, pr=False: 1 - classic_force_model(t, T=T, tau=tau, pr=pr)
            else:
                raise Exception("Something calculate_mpc_force")
            state = compute_trajectory(
                initial_state=current_state, duration=duration, force_func=force, model_type=model_type, initial_time=total_time
            )

            dist = state[1]
            if sol is None:
                sol = [dist, tau, step]
                continue

            best_dist, best_tau, best_tep = sol
            if dist > best_dist:
                sol = [dist, tau, step]

    if sol[2] == 1:
        force = lambda t, pr=False: classic_force_model(t, T=T, tau=sol[1], pr=pr)
    elif sol[2] == 2:
        force = lambda t, pr=False: 1 - classic_force_model(t, T=T, tau=sol[1], pr=pr)

    print(sol, force(t=0, pr=True), force)
    return force


def calculate_mpc_force_trivial(
    current_state: List[float],
    model_type: int,
    duration: int,
    force_encodings: List[List],
    T: float = 0.1,
    total_time=0.0,
) -> float:
    force = lambda t: classic_force_model(t, T=T, tau=(1 - 0.785))
    return force


def test_calculate_mpc_action_0():
    agent = Capsubot(dt=MIN_DT, frame_skip=1, model=1)
    n = 10
    force_encodings = []
    for tau in range(n):
        encoding = [1.0 for i in range(tau)] + [0.0 for i in range(n - tau)]
        force_encodings.append(encoding)
    action = calculate_mpc_action(
        current_state=agent.get_state,
        duration=1.0,
        force_encodings=force_encodings,
        model_type=1,
        T=0.0441,
        total_time=agent.get_total_time,
    )
    print(f"action equals to {action}")
    assert action == 1.0


if __name__ == "__main__":
    import matplotlib as plt

    test(test_caspubot_model_0)
    test(test_caspubot_model_1)
    # test(test_calculate_mpc_action_0)

    force_encodings = make_force_encodings(5)
    model_type = 1
    agent = Capsubot(dt=MIN_DT, frame_skip=10, model=model_type)
    while agent.get_total_time < 1.0:
        action = calculate_mpc_action(
            current_state=agent.get_state,
            duration=0.2,
            force_encodings=force_encodings,
            model_type=model_type,
            T=0.0441,
        )
        print(f"optimal action {action} at {agent.get_total_time}")
        agent.step(action=action)
        print(f"agent vel: {agent.get_average_speed} agent x: {agent.get_state}")
