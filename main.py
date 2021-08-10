from envs.CapsubotEnv import CapsubotEnv
import numpy as np
import time
import matplotlib.pyplot as plt

def action_law(t):
  F = 1.25
  T = 100E-3
  tau = 0.7
  tauT = T*tau
  res = t % T
  if (res < tauT):
    return F
  return 0

env = CapsubotEnv()

obst = env.reset()
for i in range(2000):
  t = env.dt*i
  action = 10*np.sin(i/np.pi)
  #action = action_law(t) // Should move with average velocity
  obs, rewards, done, info = env.step(action)
  env.render()
  time.sleep(env.dt)