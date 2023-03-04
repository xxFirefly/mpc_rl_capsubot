import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from capsubot_env.capsubot import Capsubot, classic_force_model
from capsubot_env.capsubot_env import CapsubotEnv
from collections.abc import Callable
from collections import deque
from typing import List

MIN_DT = CapsubotEnv().dt


# def compute_trajectory(
#     duration: int, force_func : Callable[[float], bool], dt: int = 0.01 / 200, model_type=0
# ) -> List[List[float]]:
#     """
#     Returns:
#         List of agent states, where state is [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
#     """
#     agent = Capsubot(dt=dt, frame_skip=1, model=model_type)
#     agent.reset()

#     first = True
#     while agent._total_time <= duration:
#         action = force_func(agent._total_time)
#         agent.step(action=action)
#     return np.append([agent.get_total_time()], agent.get_state())


def compute_classic_trajectory(
    duration: int, T: int, tau: int, dt: int = 0.01 / 200, model_type=0
) -> List[List[float]]:
    """
    Returns:
        List of agent states, where state is [time, body_position, body_velocity, inner_body_position, inner_body_velocity]
    """
    agent = Capsubot(dt=dt, frame_skip=1, model=model_type)
    agent.reset()

    states = []
    first = True
    while agent._total_time <= duration:
        if not first:
            action = classic_force_model(agent._total_time, T=T, tau=tau)
            agent.step(action=action)
        states.append(np.append([agent._total_time], agent.get_state))
        first = False
    return states


def compute_classic_trajectory_dimless(
    n_periods: int, T: int, tau: int, model_type=0, dim_dt=0.01/200
) -> List[List[float]]:
    """
    Returns:
        List of agent dimmensionless states, where state is [time, body_position, body_velocity, inner_body_position, inner_body_velocity]

    """
    dummy_agent = Capsubot(dt=0.1, frame_skip=1, model=model_type)
    dim_T = T / dummy_agent.omega
    states = compute_classic_trajectory(
        duration=dim_T * n_periods,
        T=dim_T,
        tau=tau,
        dt=dim_dt,
        model_type=model_type,
    )

    dimless_states = []
    for state in states:
        dimless_states.append(
            [
                state[0] * dummy_agent.omega,
                state[1] / dummy_agent.L,
                state[2] / dummy_agent.L / dummy_agent.omega,
                state[3] / dummy_agent.L,
                state[4] / dummy_agent.L / dummy_agent.omega,
            ]
        )
    return dimless_states


def compute_optimal_force():
    pass


def compute_metric():
    pass

def test_caspubot_model_0():
    kEps = 0.1
    states = compute_classic_trajectory_dimless(n_periods=50.0, T=1.1 * 2 * np.pi, tau=0.1, model_type=0, dim_dt=MIN_DT)
    av_velocity = states[-1][1] / states[-1][0]
    target_av_velocity = 0.047
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps


def test_caspubot_model_1():
    kEps = 0.1
    states = compute_classic_trajectory_dimless(n_periods=50.0, T=7.95, tau=0.215, model_type=1, dim_dt=MIN_DT)
    av_velocity = states[-1][1] / states[-1][0]
    target_av_velocity = 0.1868
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps

    states = compute_classic_trajectory(duration=2.0, T=0.0441, tau=0.785, dt=MIN_DT, model_type=1)
    av_velocity = states[-1][1] / states[-1][0]
    target_av_velocity = -0.0748
    assert abs((av_velocity - target_av_velocity) / target_av_velocity) < kEps

def test(func):
    func()
    print(f"{func.__name__} is OK")

# Generate all possible forces with fixed number of switches
def add_switch(value : int, array : list, max_len : int, arrays : List[List]):
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

def make_force_encodings(num_switches : int):
    force_encodings = []
    temp = []
    add_switch(-1, temp, num_switches, force_encodings)
    return force_encodings


if __name__ == "__main__":
    test(test_caspubot_model_0)
    test(test_caspubot_model_1)
