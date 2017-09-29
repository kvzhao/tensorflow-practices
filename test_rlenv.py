import numpy as np
from rlenv.envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()

print ('Env: CliffWalking')
print ('Observation space: {}'.format(env.observation_space.n))
print ('Action space: {}'.format(env.action_space.n))

state = env.reset()

print ('Random Policy')
total_reward = 0
for _ in range(20):
    action = np.random.randint(4)
    next_state, reward, done, _ = env.step(action)
    print ('St: {}, At: {}, R: {}, St+1: {}'.format(state, action, reward, next_state))
    state = next_state
    total_reward += reward
    if (done):
        print ('Done. Total reward = {}'.format(total_reward))
        break
