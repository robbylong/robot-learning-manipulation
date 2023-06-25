from rpl_pybullet_sample_env.pybullet_rl_envs.franka_gym_env import FrankaPandaEnv
import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

env = gym.make("PandaReacherLimitZ-v0", use_ee=False, controlMode='position', simMode ='gui')

print(f"Skip Steps: {env.skipSteps}")
#print(f"observation space: {env.observation_space}")
EP_STEPS = 100
NUM_EPISODES = 1 # what is this in seconds?

s = env.reset()
actions = []
states = [] 
vels = []
diffs = []
actual = []
idx = 0

prev_eef_pose_pos = env.compute_forward_kinematics(env.get_observation()[:7]).flatten()
for i in range(EP_STEPS*NUM_EPISODES):
    
    a = env.action_space.sample()
    #a[0] = np.random.uniform(0,env.rad_step)
    actions.append(a[idx]+s[idx])
    s_,r,_,actual_a = env.step(a)
    states.append(s_[idx])
    vels.append(s_[idx+env.DOF])
    diffs.append(s_[idx]-s[idx])
    s = s_
    print(f"rewards: {r} @ step: {i}")
    eef_pose_pos = env.compute_forward_kinematics(env.get_observation()[:7]).flatten()
    p.addUserDebugLine(eef_pose_pos , prev_eef_pose_pos , lineColorRGB=[1, 0.16, 0.02], lineWidth=2.0, physicsClientId=env.clientId)
    prev_eef_pose_pos  = eef_pose_pos 

    time.sleep(1.) # human watchable
fig = plt.figure()
plt.plot(actions)
plt.plot(states)
plt.plot(vels)
# plt.ylabel("states")
plt.xlabel("time")
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(states,actions,diffs)
# ax.set_xlabel('x')
# ax.set_ylabel('a')
# ax.set_zlabel('diff')
# plt.plot(actions)
# plt.plot(states)
# plt.plot(diffs)
# plt.ylabel(f'Joint {idx} angle')
plt.legend(['action','state','velocity'])
# plt.xlabel("time")
#plt.scatter(actions,diffs)
plt.savefig("state_action_plot")
env.close()
print()
#TODO: find a way to plot published commands and received commands (i.e., how much does state change by if we publish certain torque or velocity)
assert isinstance(env, gym.Env), "Could not load gym env!"