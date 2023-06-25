# Trajectory tracking with velocity control
from itertools import cycle

from matplotlib import projections
from rpl_pybullet_sample_env.pybullet_rl_envs.franka_gym_env import FrankaPandaEnv
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
env = gym.make("PandaReacherLimitZ-v0", use_ee=True, controlMode='velocity', simMode ='gui')
p.setAdditionalSearchPath(pd.getDataPath()) 
l = np.array(env.action_space.low)
h = np.array(env.action_space.high) 
print(f"action space low: {l}")
print(f"action space high: {h}")

# return joint information for all joint states (including fixed joints)
def getJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques

# return joint information for the motorized joints only
def getMotorJointStates(robot):
  joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
  joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
  joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
  joint_positions = [state[0] for state in joint_states]
  joint_velocities = [state[1] for state in joint_states]
  joint_torques = [state[3] for state in joint_states]
  return joint_positions, joint_velocities, joint_torques

def circle3D(t, p, returnVel=True):
    """
    Given t, returns a 3D position (x,y,z) and  that lies on a circle in cartesian space 
    """
    # TODO: shift the circle by the radius so that p is on the circle!
    p = np.array(p) # intitial end effector
    r = .2
    v1 = np.array([1,0,0])
    v2 = np.array([0,1,0])
    
    pos = p - np.array([r,0,0]) + r*np.cos(t)*v1 + r*np.sin(t)*v2
    if returnVel:
        zero_shift = -np.pi/2.0
        vel = -r*np.sin(t+zero_shift)*v1 + r*np.cos(t+zero_shift)*v2 - np.array([r,0,0]) 
        return  pos, vel
    return pos

# get end effector eef_pose_pos 
panda_id = env.rpl_panda.id 
numJoints = p.getNumJoints(panda_id)
PandaEndEffectorIndex = 9 # finger joints not at the end of kinematic tree

# TODO: Check that this eef_pose_pos  corresponds to eef_pose_pos  of link at index: PandaEndEffectorLink
eef_pose_pos, eef_pose_ori = env.get_ee_pose_pybullet()
startPos = eef_pose_pos # (and avoid initial jump)
initJoints = env.get_observation()[:7]


s = env.reset()

lims = [87,87,87,87,12,12,12]+[20,20]
#time.sleep(100)
vel_ = []
pos_ = []
actual_ = []
vels = []
states = []

D_gain = 0.2
P_gain = 2.0
traj_times = np.linspace(0.0,2*np.pi + np.pi,TOTAL_SIM_STEPS)
t_start = time.time()
alpha = 1.5 # controls trajectory velocity
waypoint_duration = 7500*PHYSICS_TIME_STEP

for t in range (TOTAL_SIM_STEPS):

    v = traj_times[t]/alpha
    # get desired values from circle trajectory
    desired_pos,desired_vel = circle3D(v, startPos)
    vel_.append(desired_vel)
    pos_.append(desired_pos)

    # visualize and save eef eef_pose_pos 
    eef_pose_pos, eef_pose_ori  = env.get_ee_pose_pybullet()
    actual_.append(eef_pose_pos )
    if t>0: 
        p.addUserDebugLine(eef_pose_pos , prev_eef_pose_pos , lineColorRGB=[1, 0.16, 0.02], lineWidth=2.0, lifeTime=waypoint_duration)
    prev_eef_pose_pos  = eef_pose_pos 
        
    # Get the current joint and link state directly from Bullet. (q, q., q..)
    pos, vel, torq = getJointStates(panda_id)
    mpos, mvel, mtorq = getMotorJointStates(panda_id)

    states.append(mpos)
    result = p.getLinkState(panda_id,
                            PandaEndEffectorIndex,
                            computeLinkVelocity=1,
                            computeForwardKinematics=1)
    link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

    # Get the Jacobians for the CoM of the end-effector link.
    # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    # The localPosition is always defined in terms of the link frame coordinates.

    zero_vec = [5.0] * len(mpos) # base of robot is fixed
    # compute Jacobian
    jac_t, jac_r = p.calculateJacobian(panda_id, PandaEndEffectorIndex, com_trn, mpos, zero_vec, zero_vec)

    jac_matrix = np.array(jac_t)

    # compute psuedo-inverse Jacobian
    pinv_jac = np.linalg.pinv(jac_matrix)

    # compute control 
    PD_error = D_gain*(desired_vel-link_vr) + P_gain*(desired_pos-link_trn)
    orn = p.getQuaternionFromEuler([math.pi,0., 0.])
    q_dot = np.matmul(pinv_jac, PD_error)
    vels.append(q_dot)

    # apply control
    # TODO: use env.step for this
    for i in range(env.DOF): # the first 7 joints correspond to the arm joints
        p.setJointMotorControl2(panda_id, env.rpl_panda.joint_ids[i], 
                            p.VELOCITY_CONTROL, targetVelocity=q_dot[i], 
                            force=lims[i])   
    p.stepSimulation()

times = np.linspace(0,SIM_DURATION,TOTAL_SIM_STEPS)
plt.plot(times,vels)
plt.legend([str(x) for x in list(range(env.DOF))])
plt.ylabel("joint velocity")
plt.xlabel("time (s)")
plt.show()
plt.pause(5)
plt.close()

pos = np.vstack(pos_)
vel = np.vstack(vel_)
actual = np.vstack(actual_)

plt.plot(times,vel)
plt.ylabel("eef velocity")
plt.xlabel("time (s)")
plt.show()
plt.pause(5)
plt.close()


ax = plt.axes(projection='3d')
ax.scatter(pos[:,0],pos[:,1], pos[:,2])
ax.scatter(actual[:,0], actual[:,1], actual[:,2])
ax.scatter(vel[:,0], vel[:,1], vel[:,2]) 
plt.legend(['desired_trajectory', 'actual_trajectory', 'desired_vel'])
ax.set_zlabel("World Frame Z")
ax.set_xlabel("World Frame X")
ax.set_ylabel("World Frame Y")
plt.show()
np.save('pos_trajectory_latest.npy',np.array(states))
np.save('vel_trajectory.npy',np.array(vels))
