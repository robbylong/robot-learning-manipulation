# Trajectory tracking with position control
from itertools import cycle

from matplotlib import projections
from rpl_pybullet_sample_env.src.franka_gym_env import FrankaPandaEnv
import gym
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import pybullet as p
import pybullet_data as pd

# Safety Padding (stops joint controller from going too close to joint limits)
# Only use with use_ee=False!

PHYSICS_TIME_STEP = 1/240. # controls how good the physics approximation is 
# it controls the accuracy of the collisions, object positions, etc. (we don't touch this)
# better to use high frequency for the Euler integration - Valerio

# the time step for the task, needs to be built around this. 
SIM_DURATION = 10 # time in seconds
TOTAL_SIM_STEPS = int(SIM_DURATION/PHYSICS_TIME_STEP) # seconds

# is this also about how long we wait and let a force take effect?
env = gym.make("FrankaPandaReach-v0", use_ee=False, controlMode='position', simMode ='direct', timeStep=PHYSICS_TIME_STEP)
p.setAdditionalSearchPath(pd.getDataPath()) 
l = np.array(env.action_space.low) #+ PADDING
h = np.array(env.action_space.high) #- PADDING
print(f"action space low: {l}")
print(f"action space high: {h}")

def circle3D(t, p):
    # TODO: shift the circle by the radius so that p is on the circle!
    p = np.array(p) # intitial end effector
    r = .2
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    
    return p - np.array([r,0,-0.06]) + r*np.cos(t)*v1 + r*np.sin(t)*v2

# using real robot start state
#initJoints = [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.785]

# get ee pose
pose = env.rpl_panda.get_eef_pose()
startPos = pose[:3] # (and avoid initial jump)
initOrientation = p.getQuaternionFromEuler(pose[4:])#([0,0,0])

print("resetting env..")
time.sleep(2.0)
s = env.reset()[:7]
time.sleep(2.0)
print("resetting env done...")
# for i in range(50):
#     p.loadURDF("sphere_small.urdf", circle3D(next(T),p=startPos), initOrientation, globalScaling=0.2)
initJoints,_ = env.get_observation()

# time.sleep(100)

target_ = []
actual_ = []
actions = []
states = [] 

traj_time = 0.0
t_start = time.time()
alpha = 1.6 # controls trajectory velocity

for i in range (TOTAL_SIM_STEPS):

    traj_time += PHYSICS_TIME_STEP
    states.append(s)
    # replace this with analytical equation, and query it
    v = traj_time/alpha
    pos = circle3D(v, p=startPos)
    orn = p.getQuaternionFromEuler([math.pi,0., 0.])
    a = p.calculateInverseKinematics(1, 9, pos, orn, 
                                                l, h, list(range(7)) , 
                                                initJoints, maxNumIterations=10)
    actions.append(a)
    s,r,_,_ = env.step(a)
    target_.append(pos)
    actual_.append(env.rpl_panda.get_eef_pose())
    time.sleep(1/60.)

print(f"runtime: {time.time()-t_start} seconds")

# times = np.linspace(0,SIM_DURATION,TOTAL_SIM_STEPS)
# plt.plot(times,states)
# plt.legend([str(x) for x in list(range(7))])
# plt.ylabel("joint angle")
# plt.xlabel("time")
targets = np.vstack(target_)
actuals = np.vstack(actual_)

ax = plt.axes(projection='3d')
ax.scatter(targets[:,0],targets[:,1], targets[:,2])
ax.scatter(actuals[:,0],actuals[:,1],actuals[:,2]) 
plt.legend(['target', 'actual'])
plt.show()

# plt.plot(times,actions)
# plt.plot(times,states)
# plt.ylabel(f'Joint angle')
# plt.legend(['action', 'state'])
# plt.xlabel("time")
plt.show()
env.close()
# print()
# #TODO: find a way to plot published commands and received commands (i.e., how much does state change by if we publish certain torque or velocity)
# assert isinstance(env, gym.Env), "Could not load gym env!"

np.save('trajectory_position.npy',np.array(states))