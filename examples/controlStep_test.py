import os
import numpy as np
from matplotlib import pyplot as plt 
import time

import pybullet as p
from rpl_pybullet_sample_env.src.pybullet_rpl_robots.pandas import RPL_Panda, PyBullet_Panda

PATH_TO_RPL_PANDA = os.path.dirname(__file__)+\
                    "/../rpl_panda_with_rs/urdf/panda_with_rs_PYBULLET.urdf"

# start pybullet client
clientId = p.connect(p.DIRECT)

SIM_TIME_STEP = 1/240.
p.setTimeStep(SIM_TIME_STEP, physicsClientId=clientId)

# load franka_panda robot from rpl_pybulletenv.
initOrientation = p.getQuaternionFromEuler([0,0,0])
startPos = [1,0,0] # base cartesian position
initJointPositions = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.785, 0.02, 0.02], "float32")

DOF = 7 # 7 DOF starting joint pos
# path for pykin - collision checking lib
root_path = os.path.dirname(__file__)+"/"

# load robot model
rpl_panda = RPL_Panda(startPos, initOrientation, 
                    init_joints = initJointPositions,
                    sim_camera=False, 
                    physicsClient=clientId,
                    use_ghost=True,
                    root_path=root_path)

# Simulation Loop
DURATION = 10 # seconds

TOTAL_SIM_STEPS = int(DURATION/SIM_TIME_STEP)

# same as start but rotate joint 0 by 1 radian
desired_positions = initJointPositions.copy()
desired_positions[0] = 1.0

# read initial state (this should be identical to initJointPositions)
state, velocity = rpl_panda.get_joint_state(DOF=DOF)
assert all(state == initJointPositions[:DOF]),\
     f"expected {initJointPositions[:DOF]}, but got {state}"

print(f"init velocities: {velocity}")

# state+vel containers
states = np.zeros((TOTAL_SIM_STEPS, DOF))
velocities = np.zeros((TOTAL_SIM_STEPS, DOF))

# effort limits     
el = rpl_panda.arm_el+rpl_panda.gripper_el

for tt in range(TOTAL_SIM_STEPS):
    # apply control
    # rpl_panda.apply_joint_control(controls=desired_positions, DOF=DOF, mode=p.POSITION_CONTROL)
    for i in range(DOF): # the first 7 joints correspond to the arm joints
        p.setJointMotorControl2(rpl_panda.id, rpl_panda.joint_ids[i], 
                            p.POSITION_CONTROL, desired_positions[i],
                            force=el[i],
                            maxVelocity=1.0)

    # step simulator
    p.stepSimulation(physicsClientId=clientId)

    # read next state
    state, velocity = rpl_panda.get_joint_state(DOF=DOF)
    states[tt,:] = state
    velocities[tt,:] = velocity

f, ax_list = plt.subplots(7, 2,  figsize=(15, 15), sharex=True)
legend = []
y = np.array([-np.pi, np.pi])
one_sec = np.ones_like(y)*(1/SIM_TIME_STEP)
for d in range(DOF):
    ax_list[d,0].plot(states[:,d],f"C{d}")
    ax_list[d,0].plot(one_sec,y,'C7')
    ax_list[d,0].set_ylim([-np.pi,np.pi])
    ax_list[d,0].legend([f"joint state {d}", "time=1.0 sec"])

    plt.plot(one_sec,y,'C7')
    ax_list[d,1].plot(velocities[:,d],f"C{d}")
    ax_list[d,1].plot(one_sec,y,'C7')
    ax_list[d,1].set_ylim([-np.pi,np.pi])
    ax_list[d,1].legend([f"joint vel {d}", "time=1.0 sec"])





ax_list[-1,0].set_xlabel('time')
ax_list[-1,1].set_xlabel('time')
# plt.ylabel("joint angle")
plt.savefig("pid_test",dpi=300)