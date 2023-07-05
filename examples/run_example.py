USE_ROSPY = False
N_SIM_STEPS = 50000
STEP_FREQUENCY = 240*10*2

import time
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import deque


#load object models
import object_models
import os
import inspect

#import affcor
from UCL_AffCorrs.demos.show_part_annotation_correspondence_pybullet import ShowPartCorrespondence
import cv2

#import RL policy
from explore_policy_net import ExplorePolicyNet

OBJECT_PATH = os.path.dirname(inspect.getfile(object_models))

# Adding pybullet files. 
import pybullet as p
import pybullet_data as pd
# p.GUI for GUI or p.DIRECT for non-graphical version
# e.g for when you're training on an HPC or distributed

# User-defined constants
SUPPORT_DIR = "../src/UCL_AffCorrs/affordance_database/temp/"

physicsClient = p.connect(p.GUI)
# Add path to pandas package.
#import src.rpl_pybullet_sample_env
from rpl_pybullet_sample_env.pybullet_robots.arms.panda import RPL_Panda, PyBullet_Panda, PandaWithCustomGripper
#import pybullet_robots.xarm.xarm_sim as xarm_sim 

#import camera and object
from pb_camera_object import Object, SDFObject, CameraObject
# Most simulations need the ground plane to be added.
# and gravity pointing down at Z-axis
p.setAdditionalSearchPath(pd.getDataPath()) 
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("plane.urdf")
# Set the time step
p.setTimeStep(1/240/10/2)

initOrientation = p.getQuaternionFromEuler([0,0,0])

fixed_first_segment = False

#######################
# Spring stiffness #
#######################
# k values
EI = 0.32
L = 0.235
N = 5
segment_length = L/N
k_1 = 2*EI/L*N*N/(N-1/3)
k_n = EI/L*N
k = [k_1,k_n,k_n,k_n,k_n]
k = k + k + k
# Desired positions for each joint segment
desired_positions_finger_1 = [0.0, 0.0, 0.0, 0.0, 0.0]
desired_positions_finger_2 = [0.0, 0.0, 0.0, 0.0, 0.0]
desired_positions_finger_3 = [0.0, 0.0, 0.0, 0.0, 0.0]
desired_positions_segment = desired_positions_finger_1+desired_positions_finger_2+desired_positions_finger_3
# Damping coefficient
damping_coefficient = 0.01
# segment joint indexes (hard coded)
segment_index_finger_1 = [12,13,14,15,16]
segment_index_finger_2 = [20,21,22,23,24]
segment_index_finger_3 = [28,29,30,31,32]
# segment_indices = segment_index_finger_1 + segment_index_finger_2 + segment_index_finger_3
segment_indices = [segment_index_finger_1,segment_index_finger_2,segment_index_finger_3]
segment_indices_row = segment_index_finger_1 + segment_index_finger_2 + segment_index_finger_3

# what are the actions? X=gripper prismatic joint, Y=gripper revolute joint, Z=gripper palm, H=gripper height
action_names = ["X_close", "X_open", "Y_close", "Y_open", "Z_close", "Z_open", "H_down", "H_up"]
##############
# Table #
##############
#table
table_height_offset = 0.625
PATH_TO_table = OBJECT_PATH +\
                    "/assets/google/table/table.urdf"

object_position = [0.3, 0, 0]  # Specify the position of the object in the world
object_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # Specify the orientation of the object

# Create an instance of the Object class to import the object
table = Object(PATH_TO_table, object_position, object_orientation)

##############
# RPL Robots #
##############
# startPos = [1,0,0]
# jointPositions=[0.0, 0.0, 0.0, -1.571, -0.0, 1.571, 0.785, 0.02, 0.02]
# rpl_panda = RPL_Panda(startPos, initOrientation, 
#                       init_joints = jointPositions,
#                       sim_camera=False,
#                       physicsClient=physicsClient)
# p.addUserDebugText("RPL Panda", [1,0,1], [1,0,0], 1, 0)    

##############
# RPL Robots with luke gripper #
##############

# startPos = [1,1,0]
startPos = [0,0,0+table_height_offset]
jointPositions=[0.0, 0.0, 0.0, -1.571, -0.0, 1.571, 0.785, 0.02, 0.02]
# replaces the default gripper with initial Luke Gripper joints
jointPositions = jointPositions[:-2] + [140e-3, 0.,0,0,0,0,0, 140e-3, 0., 0,0,0,0,0, 140e-3, 0., 0,0,0,0,0, 0.]
# jointPositions = jointPositions[:-2] + [140e-3, 0., 140e-3, 0., 140e-3, 0., 0.]     
print(jointPositions)
rpl_panda_w_gripper = PandaWithCustomGripper(startPos, initOrientation, 
                      init_joints = jointPositions,
                      sim_camera=False,
                      physicsClient=physicsClient)
p.addUserDebugText("RPL Panda with L-Gripper", [startPos[0],startPos[1],startPos[2]+1], [1,0,0], 1, 0)    

# # add spring stiffness to gripper finger segments
# for i in range(N*3):
#     # Modify the dynamics properties of the joint to include spring stiffness
#     p.changeDynamics(rpl_panda_w_gripper.id, segment_indices_row[i], jointDamping=damping_coefficient, jointStiffness=k[i])

##############
# camera #
##############

# set camera pos and ori
camera = CameraObject()
camera.pos = [0.5,0,0.7+table_height_offset]
camera.target = [0.49,0,0.6+table_height_offset]
camera.view_matrix = p.computeViewMatrix(camera.pos, camera.target, camera.axis)
camera.projection_matrix = p.computeProjectionMatrixFOV(camera.fov, camera.aspect, 
                                                        camera.near, camera.far)

time.sleep(1.) # wait 1 sec for everything to load up

# Configure debug visualizer options for texture rendering
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

motor_sign = 1
leadscrew_dist = 35e-3
def calc_y (th_rad,x):
    return x + motor_sign * leadscrew_dist * np.sin(th_rad)

def read_gauge(finger,segment_length,numJoint_per_finger,poly_degree,x_pos):

    joint_values = np.zeros(numJoint_per_finger)
    for i in range(len(joint_values)):
        joint_state = p.getJointState(rpl_panda_w_gripper.id,segment_indices[finger][i])
        joint_values[i] = joint_state[0]
    cumulative = np.zeros(numJoint_per_finger)
    finger_xy = np.zeros((numJoint_per_finger+1,2))

    #If the first segment is locked
    if fixed_first_segment:
        finger_xy[0,0] = segment_length
    else:
        finger_xy[0,0] = 0

    for i in range(numJoint_per_finger):
        # find the cumulative total of the angular sum
        if i == 0:
            cumulative[i] = joint_values[i]
        else:
            cumulative[i] = cumulative[i-1] + joint_values[i]
        
        # Calculate cartesian coordinates of each joint
        # finger_xy[i+1,0] = finger_xy[i,0] + segment_length*np.cos(cumulative[i])
        # finger_xy[i+1,1] = finger_xy[i,1] + segment_length*np.sin(cumulative[i])
        finger_xy[i+1,0] = finger_xy[i,0] + segment_length*np.cos(joint_values[i])
        finger_xy[i+1,1] = finger_xy[i,1] + segment_length*np.sin(joint_values[i])
    # print("the cartesian coordiantes of segment joints: ")
    # print(finger_xy)
    # Polyfit a cubic curve to these joint positions
    coeff = np.polyfit(finger_xy[:, 0], finger_xy[:, 1],poly_degree)

    # Evaluate y at the gauge x position
    y = 0.0
    for i in range(poly_degree + 1):
        y += coeff[i] * np.power(x_pos, poly_degree-i)

    # Return y value in millimeters, unprocessed
    return y * 1000

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def policy_input(finger_1,finger_2,finger_3,palm,wrist,motor_x,motor_y,motor_z,base_z):
    state_range = [-1,1]
    ROT_LIMIT = 0.7
    gripper_ll = [49e-3, -ROT_LIMIT, 49e-3, -ROT_LIMIT, 49e-3, -ROT_LIMIT, 0.0e-3]
    gripper_ul = [134e-3, ROT_LIMIT, 134e-3, ROT_LIMIT, 134e-3, ROT_LIMIT, 165e-3]
    bend_range = [-2.5, +2.5]
    palm_range = [-8, +8]
    wrist_range = [-5, +5]
    x_range = [gripper_ll[0], gripper_ul[0]]
    y_range = [gripper_ll[0], gripper_ul[0]]
    z_range = [gripper_ll[6], gripper_ul[6]]   
    base_range = [-30e-3, 30e-3]

    x_avg = (finger_1[0] + finger_1[1])/2
    x_t0 = reading_normalization(finger_1[0],bend_range[0],bend_range[1])
    x_t1 = reading_normalization(finger_1[1],bend_range[0],bend_range[1])
    x_avg = reading_normalization(x_avg,bend_range[0],bend_range[1])
    bending_gauge_1_normalized = [x_t0, x_avg, x_t1]

    x_avg = (finger_2[0] + finger_2[1])/2
    x_t0 = reading_normalization(finger_2[0],bend_range[0],bend_range[1])
    x_t1 = reading_normalization(finger_2[1],bend_range[0],bend_range[1])
    x_avg = reading_normalization(x_avg,bend_range[0],bend_range[1])
    bending_gauge_2_normalized = [x_t0, x_avg, x_t1]

    x_avg = (finger_3[0] + finger_3[1])/2
    x_t0 = reading_normalization(finger_3[0],bend_range[0],bend_range[1])
    x_t1 = reading_normalization(finger_3[1],bend_range[0],bend_range[1])
    x_avg = reading_normalization(x_avg,bend_range[0],bend_range[1])
    bending_gauge_3_normalized = [x_t0, x_avg, x_t1]

    x_avg = (palm[0] + palm[1])/2
    x_t0 = reading_normalization(palm[0],palm_range[0],palm_range[1])
    x_t1 = reading_normalization(palm[1],palm_range[0],palm_range[1])
    x_avg = reading_normalization(x_avg,palm_range[0],palm_range[1])
    palm_gauge_normalized = [x_t0, x_avg, x_t1]

    x_avg = (wrist[0] + wrist[1])/2
    x_t0 = reading_normalization(wrist[0],wrist_range[0],wrist_range[1])
    x_t1 = reading_normalization(wrist[1],wrist_range[0],wrist_range[1])
    x_avg = reading_normalization(x_avg,wrist_range[0],wrist_range[1])
    wrist_gauge_normalized = [x_t0, x_avg, x_t1]

    #generate motor x states (recent 6)
    motor_x_copy = motor_x[:]
    for motor in range(len(motor_x_copy)):
        motor_x_copy[motor] = reading_normalization(motor_x_copy[motor],x_range[0],x_range[1])
        # motor_x_copy[motor] = round(motor_x_copy[motor],3)
    delta = [sign(motor_x_copy[1]-motor_x_copy[0]),sign(motor_x_copy[2]-motor_x_copy[1]),sign(motor_x_copy[3]-motor_x_copy[2]),sign(motor_x_copy[4]-motor_x_copy[3]),sign(motor_x_copy[5]-motor_x_copy[4])]
    motor_x_states = [motor_x_copy[0],delta[0],motor_x_copy[1],delta[1],motor_x_copy[2],delta[2],motor_x_copy[3],delta[3],motor_x_copy[4],delta[4],motor_x_copy[5]]

    #generate motor y states (recent 6)
    motor_y_copy = motor_y[:]
    for motor in range(len(motor_y_copy)):
        motor_y_copy[motor] = reading_normalization(motor_y_copy[motor],y_range[0],y_range[1])
        # motor_y_copy[motor] = round(motor_y_copy[motor],3)
    delta = [sign(motor_y_copy[1]-motor_y_copy[0]),sign(motor_y_copy[2]-motor_y_copy[1]),sign(motor_y_copy[3]-motor_y_copy[2]),sign(motor_y_copy[4]-motor_y_copy[3]),sign(motor_y_copy[5]-motor_y_copy[4])]
    motor_y_states = [motor_y_copy[0],delta[0],motor_y_copy[1],delta[1],motor_y_copy[2],delta[2],motor_y_copy[3],delta[3],motor_y_copy[4],delta[4],motor_y_copy[5]]

    #generate motor z states (recent 6)
    motor_z_copy = motor_z[:]
    for motor in range(len(motor_z_copy)):
        motor_z_copy[motor] = reading_normalization(motor_z_copy[motor],z_range[0],z_range[1])
        # motor_z_copy[motor] = round(motor_z_copy[motor],3)
    delta = [sign(motor_z_copy[1]-motor_z_copy[0]),sign(motor_z_copy[2]-motor_z_copy[1]),sign(motor_z_copy[3]-motor_z_copy[2]),sign(motor_z_copy[4]-motor_z_copy[3]),sign(motor_z_copy[5]-motor_z_copy[4])]
    motor_z_states = [motor_z_copy[0],delta[0],motor_z_copy[1],delta[1],motor_z_copy[2],delta[2],motor_z_copy[3],delta[3],motor_z_copy[4],delta[4],motor_z_copy[5]]

    #generate base z states (recent 6)
    base_z_copy = base_z[:]
    for motor in range(len(base_z_copy)):
        base_z_copy[motor] = reading_normalization(base_z_copy[motor],base_range[0],base_range[1])
        # base_z_copy[motor] = round(base_z_copy[motor],3)
    delta = [sign(base_z_copy[1]-base_z_copy[0]),sign(base_z_copy[2]-base_z_copy[1]),sign(base_z_copy[3]-base_z_copy[2]),sign(base_z_copy[4]-base_z_copy[3]),sign(base_z_copy[5]-base_z_copy[4])]
    base_z_states = [base_z_copy[0],delta[0],base_z_copy[1],delta[1],base_z_copy[2],delta[2],base_z_copy[3],delta[3],base_z_copy[4],delta[4],base_z_copy[5]]

    policy_input_vector = bending_gauge_1_normalized + bending_gauge_2_normalized + bending_gauge_3_normalized + palm_gauge_normalized + wrist_gauge_normalized + motor_x_states + motor_y_states + motor_z_states + base_z_states
    
    return policy_input_vector
    

def reading_normalization(value, min_value, max_value):
    if value >= max_value:
        normalized_value = 1
    elif value <= min_value:
        normalized_value = -1
    else:
        normalized_value = (value - min_value) / (max_value - min_value) * 2 - 1

    return round(normalized_value,3)

def motor_state_change(action_name):
    if action_name == "X_open" or action_name == "X_close":
        if action_name == "X_open":
            motor_X_change.append(+1)
        else:
            motor_X_change.append(-1)
        motor_Y_change.append(0)
        motor_Z_change.append(0)
        base_Z_change.append(0)
    elif action_name == "Y_open" or action_name == "Y_close":
        if action_name == "Y_open":
            motor_Y_change.append(+1)
        else:
            motor_Y_change.append(-1)
        motor_X_change.append(0)
        motor_Z_change.append(0)
        base_Z_change.append(0)
    elif action_name == "Z_open" or action_name == "Z_close":
        if action_name == "Z_open":
            motor_Z_change.append(+1)
        else:
            motor_Z_change.append(-1)
        motor_X_change.append(0)
        motor_Y_change.append(0)
        base_Z_change.append(0)
    elif action_name == "H_up" or action_name == "H_down":
        if action_name == "H_up":
            base_Z_change.append(+1)
        else:
            base_Z_change.append(-1)
        motor_X_change.append(0)
        motor_Y_change.append(0)
        motor_Z_change.append(0)

# robot move aside for camera to capture image
p.enableJointForceTorqueSensor(rpl_panda_w_gripper.id,9,True)
p.enableJointForceTorqueSensor(rpl_panda_w_gripper.id,34,True)
fm_wrist_list = []
fm_palm_list = []
t = 0
N_UP_STEPS = STEP_FREQUENCY*5
for i in range (N_UP_STEPS):
    for j in range(N*3):
        joint_state = p.getJointState(rpl_panda_w_gripper.id, segment_indices_row[j])
        joint_position = joint_state[0]
        joint_velocity = joint_state[1]
        # print("joint index: " + str(segment_indices_row[j]) + str(joint_position))
        # print(k[j])
        # # print("stiffness: " + str(stiffness))
        # # Calculate the spring force based on the joint position and stiffness
        # spring_torque = k[j] * (0 - joint_position)
        # if spring_torque > 0:
        #     spring_torque -= abs(damping_coefficient*joint_velocity)
        # else:
        #     spring_torque += abs(damping_coefficient*joint_velocity)
        # print(spring_torque)
        
        # # Apply the torque to the joint
        # p.setJointMotorControl2(rpl_panda_w_gripper.id, segment_indices_row[j], p.TORQUE_CONTROL, force=spring_torque)
        # Set the motor control mode to POSITION_CONTROL with desired gains
        p.setJointMotorControl2(
            rpl_panda_w_gripper.id,
            segment_indices_row[j],
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,  # Set the desired joint position here
            # positionGain = 0.5,
            # positionGain=k[j]/10,
            # velocityGain=damping_coefficient
        )
    # Get the contact points in the simulation
    contact_points = p.getContactPoints()
    # Find the contact point associated with the robot and the specified link index
    link_contact_points = [contact for contact in contact_points if contact[1] == rpl_panda_w_gripper.id and contact[3] == 34]

    # print("The palm force is :")
    # print(appliedForcesOrTorques[2])
    p.stepSimulation() 
    # time.sleep is for us to observe. 
    # If no GUI is used, this can be sped up. 
    time.sleep(1./STEP_FREQUENCY) 

    t = t + 1./N_UP_STEPS

    rpl_panda_w_gripper.step()

    # print the wrist force (panda joint 8 fixed joint)
    jointState = p.getJointState(rpl_panda_w_gripper.id,9)
    # Extract the applied forces or torques
    appliedForcesOrTorques = jointState[2]
    # print("The wrist force is: " + str(jointState))
    # Extract the axial forces (along the joint's axis)
    axialForces = appliedForcesOrTorques[2]
    print(axialForces)
    rpl_panda_w_gripper.move_upward(t)
    

    
        

# ##############
# # Objects #
# ##############

# #drawer container
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Design_Ideas_Drawer_Store_Organizer/model.sdf"

# object_position = [0.3, 0, 0+table_height_offset]  # Specify the position of the object in the world
# object_orientation = p.getQuaternionFromEuler([0, 0, 1.5708])  # Specify the orientation of the object

# # Create an instance of the Object class to import the object
# spatula = SDFObject(PATH_TO_obj, object_position, object_orientation)

# #Hammer
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Cole_Hardware_Hammer_Black/model.sdf"

# object_position = [0.55, -0.2, 0+table_height_offset]  # Specify the position of the object in the world
# object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# # Create an instance of the Object class to import the object
# spatula = SDFObject(PATH_TO_obj, object_position, object_orientation)

# #ball
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler/model.sdf"

# object_position = [0.6, 0, 0+table_height_offset]  # Specify the position of the object in the world
# object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# # Create an instance of the Object class to import the object
# bowl = SDFObject(PATH_TO_obj, object_position, object_orientation)

# #mug
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Room_Essentials_Mug_White_Yellow/model.sdf"

# object_position = [0.6, 0.1, 0+table_height_offset+0.05]  # Specify the position of the object in the world
# object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# Create an instance of the Object class to import the object
# mug = SDFObject(PATH_TO_obj, object_position, object_orientation)

# #screwdriver
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Nintendo_Mario_Action_Figure/model.sdf"

# object_position = [0.6, 0.2, 0+table_height_offset+0.09]  # Specify the position of the object in the world
# object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# # Create an instance of the Object class to import the object
# mario = SDFObject(PATH_TO_obj, object_position, object_orientation)

#coke can
# PATH_TO_obj = OBJECT_PATH +\
#                     "/assets/google/Room_Essentials_Bowl_Turquiose/model.sdf"
PATH_TO_obj = OBJECT_PATH +\
                    "/assets/google/Coke/model.sdf"

object_position = [0.5, -0.2, 0+table_height_offset]  # Specify the position of the object in the world
object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# Create an instance of the Object class to import the object
coke = SDFObject(PATH_TO_obj, object_position, object_orientation)

#meshes
PATH_TO_obj = OBJECT_PATH +\
                    "/assets/google/Cole_Hardware_Antislip_Surfacing_Material_White/model.sdf"

object_position = [0.6, 0, 0+table_height_offset+0.1]  # Specify the position of the object in the world
object_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Specify the orientation of the object

# Create an instance of the Object class to import the object
tissue = SDFObject(PATH_TO_obj, object_position, object_orientation)

# # ##############
# # # camera capture #
# # ##############

# p.stepSimulation() 
# # # Capture camera data
# rgb, depth_buffer_tiny, depth_tiny, sem = camera.capture()

# ##############
# # Affcorrs find part #
# ##############

# # Save the RGB image as a JPG file
# rgb = rgb[:,:,:3]
# cv2.imwrite('rgb_image.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
# affcor = ShowPartCorrespondence()
# affcor.run_result(rgb,SUPPORT_DIR)
# # print(rgb/255)
# # print(rgb.shape)
# pc = camera.get_point_cloud(depth_buffer_tiny)
# mask_points = pc[affcor.part_out_i.squeeze()]
# mask_rgb = rgb[affcor.part_out_i.squeeze()]
# print(mask_points.shape)
# print(mask_rgb.shape)
# # print(gpc)
# print(pc.shape)
# normalized_rgb = rgb/255.0
# # print(normalized_rgb_without_alpha.shape)
# normalized_rgb = normalized_rgb.reshape(-1,3)
# normalized_mask_rgb = mask_rgb/255.0
# pc = pc.reshape(-1,3)
# # print(gpc)
# pcd = camera.make_pointcloud(pc,normalized_rgb)
# pcd_mask = camera.make_pointcloud(mask_points,normalized_mask_rgb)

# # find the centre pos of part point cloud 
# max_x = max(np.asarray(pcd_mask.points)[:,0])
# max_y = max(np.asarray(pcd_mask.points)[:,1])
# max_z = max(np.asarray(pcd_mask.points)[:,2])
# min_x = min(np.asarray(pcd_mask.points)[:,0])
# min_y = min(np.asarray(pcd_mask.points)[:,1])
# min_z = min(np.asarray(pcd_mask.points)[:,2])
# obj_x = (max_x+min_x)/2
# obj_y = (max_y+min_y)/2
# obj_z = (max_z+min_z)/2
# print(obj_x)
# print(obj_y)
# print(obj_z)

# # visualize the full point cloud and part point cloud
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd_mask])

# obj_x = 0.385878
# obj_y = -0.114797
# obj_x = 0.5
# obj_y = -0.2
obj_x = 0.6
obj_y = 0
# obj_x = 0.55
# obj_y = -0.3
obj_z = 0.701805
t = 0
N_UP_STEPS = STEP_FREQUENCY*2
for i in range (N_UP_STEPS):
    for j in range(N*3):
        joint_state = p.getJointState(rpl_panda_w_gripper.id, segment_indices_row[j])
        joint_position = joint_state[0]
        joint_velocity = joint_state[1]
        print("joint position: ",joint_position)
        # print("joint index: " + str(joint_index))
        # print("stiffness: " + str(stiffness))
        # Calculate the spring force based on the joint position and stiffness
        spring_torque = k[j] * (0 - joint_position)
        if spring_torque > 0:
            spring_torque -= abs(damping_coefficient*joint_velocity)
        else:
            spring_torque += abs(damping_coefficient*joint_velocity)
        # print(spring_torque)
        # Apply the torque to the joint
        p.setJointMotorControl2(rpl_panda_w_gripper.id, segment_indices_row[j], p.TORQUE_CONTROL,force=spring_torque)

    p.stepSimulation() 
    # time.sleep is for us to observe.
    # If no GUI is used, this can be sped up. 
    time.sleep(1./STEP_FREQUENCY) 

    t = t + 1./N_UP_STEPS

    rpl_panda_w_gripper.step()
    rpl_panda_w_gripper.move_to_above(t,obj_x,obj_y,obj_z)
    print(rpl_panda_w_gripper.get_base_state())
    # rpl_panda_w_gripper.move_to_above(t,obj_x,obj_y,table_height_offset+0.01)    

t = 0
N_UP_STEPS = STEP_FREQUENCY*3
for i in range (N_UP_STEPS):
    for j in range(N*3):
        joint_state = p.getJointState(rpl_panda_w_gripper.id, segment_indices_row[j])
        joint_position = joint_state[0]
        joint_velocity = joint_state[1]
        print("joint position: ",joint_position)
        # print("joint index: " + str(joint_index))
        # print("stiffness: " + str(stiffness))
        # Calculate the spring force based on the joint position and stiffness
        spring_torque = k[j] * (0 - joint_position)
        if spring_torque > 0:
            spring_torque -= abs(damping_coefficient*joint_velocity)
        else:
            spring_torque += abs(damping_coefficient*joint_velocity)
        # print(spring_torque)
        # Apply the torque to the joint
        p.setJointMotorControl2(rpl_panda_w_gripper.id, segment_indices_row[j], p.TORQUE_CONTROL,force=spring_torque)

    p.stepSimulation() 
    # time.sleep is for us to observe.
    # If no GUI is used, this can be sped up. 
    time.sleep(1./STEP_FREQUENCY) 

    t = t + 1./N_UP_STEPS

    rpl_panda_w_gripper.step()
    rpl_panda_w_gripper.move_to_object(t,obj_x,obj_y,obj_z)
    print(rpl_panda_w_gripper.get_base_state())
    # rpl_panda_w_gripper.move_to_above(t,obj_x,obj_y,table_height_offset+0.01)  
    

time.sleep(3)

# simulation of grasp (incomplete)
t = 0
# policy object
policy = ExplorePolicyNet()

# initial states
bending_gauge_1 = deque(maxlen=2)
bending_gauge_2 = deque(maxlen=2)
bending_gauge_3 = deque(maxlen=2)

palm_gauge = deque(maxlen=2)
wrist_gauge = deque(maxlen=2)

motor_state_X = deque(maxlen=6)
motor_state_Y = deque(maxlen=6)
motor_state_Z = deque(maxlen=6)
base_state_Z = deque(maxlen=6)

motor_X_change = deque(maxlen=5)
motor_Y_change = deque(maxlen=5)
motor_Z_change = deque(maxlen=5)
base_Z_change = deque(maxlen=5)

for i in range(5):
    motor_X_change.append(0)
    motor_Y_change.append(0)
    motor_Z_change.append(0)
    base_Z_change.append(0)

t = 0
N_UP_STEPS = STEP_FREQUENCY*20
for step in range (N_UP_STEPS):
    # for j in range(N*3):
        # joint_state = p.getJointState(rpl_panda_w_gripper.id, segment_indices_row[j])
        # joint_position = joint_state[0]
        # joint_velocity = joint_state[1]
        # # print(joint_position)
        # # print("joint index: " + str(joint_index))
        # # print("stiffness: " + str(stiffness))
        # # Calculate the spring force based on the joint position and stiffness
        # spring_torque = k[j] * (0 - joint_position)     
        # if spring_torque > 0:
        #     spring_torque -= abs(damping_coefficient*joint_velocity)
        # else:
        #     spring_torque += abs(damping_coefficient*joint_velocity)
        # # print(spring_torque)
        
        # # Apply the torque to the joint
        # p.setJointMotorControl2(rpl_panda_w_gripper.id, segment_indices_row[j], p.TORQUE_CONTROL, force=spring_torque)
        # p.setJointMotorControl2(
        #     rpl_panda_w_gripper.id,
        #     segment_indices_row[j],
        #     controlMode=p.POSITION_CONTROL,
        #     targetPosition=0.0,  # Set the desired joint position here
        #     # positionGain = 0.01,
        #     # positionGain=k[j],
        #     # velocityGain=damping_coefficient
        # )

    y = read_gauge(0,segment_length,N,3,0.235-50e-3)
    bending_gauge_1.append(y)
    y = read_gauge(1,segment_length,N,3,0.235-50e-3)
    bending_gauge_2.append(y)
    y = read_gauge(2,segment_length,N,3,0.235-50e-3)
    bending_gauge_3.append(y)

    # print the wrist force (panda joint 8 fixed joint)
    jointState = p.getJointState(rpl_panda_w_gripper.id,9)
    # Extract the applied forces or torques
    appliedForcesOrTorques = jointState[2]
    # print("The wrist force is: " + str(jointState))
    # Extract the axial forces (along the joint's axis)
    # axialForces = appliedForcesOrTorques[2] + 27.72
    # axialForces = appliedForcesOrTorques[2]
    axialForces = appliedForcesOrTorques[2] - 23.3
    # axialForces = appliedForcesOrTorques[2]
    wrist_gauge.append(-axialForces)

    # # print the palm force (panda joint 8 fixed joint)
    # jointState = p.getJointState(rpl_panda_w_gripper.id,34)
    # # Extract the applied forces or torques
    # appliedForcesOrTorques = jointState[2]
    # # print("The palm force is: " + str(appliedForcesOrTorques))
    # # Extract the axial forces (along the joint's axis)
    # axialForces = appliedForcesOrTorques[2]
    # print(appliedForcesOrTorques)
    # palm_gauge.append(axialForces)

    # # Get the contact points in the simulation
    contact_points = p.getContactPoints(rpl_panda_w_gripper.id, tissue.id, 34)
    palm_force = 0
    for contact in contact_points:
        print("the contact i is: ")
        print(contact)
        palm_force += contact[9]*contact[7][2]
    palm_gauge.append(palm_force)
    # # Find the contact point associated with the robot and the specified link index
    # link_contact_points = [contact for contact in contact_points if contact[2] == rpl_panda_w_gripper.id and contact[4] == 34]

    # # Calculate the total contact force on the point of interest
    # total_contact_force = [0, 0, 0]
    # for contact in link_contact_points:
    #     contact_normal = contact[7]  # Normal vector of the contact
    #     contact_force = contact[9]  # Total force applied at the contact point
    #     total_contact_force[0] += contact_force * contact_normal[0]
    #     total_contact_force[1] += contact_force * contact_normal[1]
    #     total_contact_force[2] += contact_force * contact_normal[2]
    # palm_force = 0
    # for contact in link_contact_points:
    #     palm_force += contact[-2]
    # print("the total contact force is: " + str(palm_force))
    # palm_gauge.append(-palm_force)
    # palm_gauge.append("the total contact force is: " + str(palm_force))

    # assert p.getJointState(rpl_panda_w_gripper.id,10) == p.getJointState(rpl_panda_w_gripper.id,18) == p.getJointState(rpl_panda_w_gripper.id,26), "The gripper prismatic joints are not equal!"
    X_state = (p.getJointState(rpl_panda_w_gripper.id,10)[0] + p.getJointState(rpl_panda_w_gripper.id,18)[0] + p.getJointState(rpl_panda_w_gripper.id,26)[0])/3
    # X_state = X_state[0]
    # X_state = rpl_panda_w_gripper.grip_pris
    motor_state_X.append(X_state)
    # print("the motor X state: " + str(rpl_panda_w_gripper.grip_pris))
    # rpl_panda_w_gripper.grip_pris = X_state

    # assert p.getJointState(rpl_panda_w_gripper.id,11) == p.getJointState(rpl_panda_w_gripper.id,19) == p.getJointState(rpl_panda_w_gripper.id,27), "The gripper revolute joints are not equal!"
    Y_state = (p.getJointState(rpl_panda_w_gripper.id,11)[0]+p.getJointState(rpl_panda_w_gripper.id,19)[0]+p.getJointState(rpl_panda_w_gripper.id,27)[0])/3
    # Y_state = Y_state[0]
    # Y_state = rpl_panda_w_gripper.grip_rot
    Y_state = calc_y(Y_state,X_state)
    motor_state_Y.append(Y_state)
    # print(Y_state)
    # print("the motor Y state: " + str(rpl_panda_w_gripper.grip_rot))
    # rpl_panda_w_gripper.grip_rot = Y_state

    Z_state = p.getJointState(rpl_panda_w_gripper.id,34)
    Z_state = Z_state[0]
    # Z_state = rpl_panda_w_gripper.grip_palm
    motor_state_Z.append(Z_state)
    # print("the motor Z state: " + str(rpl_panda_w_gripper.grip_palm))
    # rpl_panda_w_gripper.grip_palm = Z_state

    H_state = rpl_panda_w_gripper.get_base_state()
    base_state_Z.append(H_state) 
    # rpl_panda_w_gripper.base_height = H_state
    # print(H_state)

    if step > 4:
        print("bending_gauge_1: " + ", ".join(str(value) for value in list(bending_gauge_1)))
        print("bending_gauge_2: " + ", ".join(str(value) for value in list(bending_gauge_2)))
        print("bending_gauge_3: " + ", ".join(str(value) for value in list(bending_gauge_3)))
        print("wrist_sensor: " + ", ".join(str(value) for value in list(wrist_gauge)))
        print("palm_sensor: " + ", ".join(str(value) for value in list(palm_gauge)))
        print("motor_state_X: " + ", ".join(str(value) for value in list(motor_state_X)))
        print("motor_state_Y: " + ", ".join(str(value) for value in list(motor_state_Y)))
        print("motor_state_Z: " + ", ".join(str(value) for value in list(motor_state_Z)))
        print("base_state_Z: " + ", ".join(str(value) for value in list(base_state_Z)))

        policy_input_vector = policy_input(bending_gauge_1,bending_gauge_2,bending_gauge_3,palm_gauge,wrist_gauge,list(motor_state_X),list(motor_state_Y),list(motor_state_Z),list(base_state_Z))
        print(policy_input_vector)
        action_number = policy.run_policy(np.array(policy_input_vector))
        # motor_state_change(action_names[action_number])
        # print("motor_X_change: ")
        # print(motor_X_change)
        # print("motor_Y_change: ")
        # print(motor_Y_change)
        # print("motor_Z_change: ")
        # print(motor_Z_change)
        # print("base_z_change: ")
        # print(base_Z_change)
        policy_t = 0
        policy_steps = 500
        for policy_step in range(policy_steps):
            # for j in range(N*3):
            #     # Apply the torque to the joint
            #     p.setJointMotorControl2(
            #     rpl_panda_w_gripper.id,
            #     segment_indices_row[j],
            #     controlMode=p.POSITION_CONTROL,
            #     targetPosition=0.0,  # Set the desired joint position here
            #     # positionGain = 0.01,
            #     # positionGain=k[j],
            #     # velocityGain=damping_coefficient
            #     )
            for j in range(N*3):
                joint_state = p.getJointState(rpl_panda_w_gripper.id, segment_indices_row[j])
                joint_position = joint_state[0]
                joint_velocity = joint_state[1]
                # print(joint_position)
                # print("joint index: " + str(joint_index))
                # print("stiffness: " + str(stiffness))
                # Calculate the spring force based on the joint position and stiffness
                spring_torque = k[j] * (0 - joint_position)     
                if spring_torque > 0:
                    spring_torque -= abs(damping_coefficient*joint_velocity)
                else:
                    spring_torque += abs(damping_coefficient*joint_velocity)
                # print(spring_torque)
                
                # Apply the torque to the joint
                p.setJointMotorControl2(rpl_panda_w_gripper.id, segment_indices_row[j], p.TORQUE_CONTROL, force=spring_torque)
            p.stepSimulation() 
            # time.sleep is for us to observe. 
            # If no GUI is used, this can be sped up. 
            time.sleep(1./STEP_FREQUENCY/policy_steps)
            policy_t = policy_t + 1./policy_steps
            rpl_panda_w_gripper.step()
            rpl_panda_w_gripper.grasp(policy_t, action_names[action_number],obj_x,obj_y,obj_z)
            # rpl_panda_w_gripper.grasp(policy_t, "Y_close",obj_x,obj_y,obj_z)

        time.sleep(1./STEP_FREQUENCY) 

        t = t + 1./STEP_FREQUENCY
    # else:
    #     p.stepSimulation() 
    #     # time.sleep is for us to observe. 
    #     # If no GUI is used, this can be sped up. 
    #     time.sleep(1./STEP_FREQUENCY) 

    #     t = t + 1./STEP_FREQUENCY

    #     rpl_panda_w_gripper.step()
    
    
    
time.sleep(100)
# Disconnect the simulation at the end of your run
p.disconnect() 
# import matplotlib.pyplot as plt
# list_size = len(fm_wrist_list)
# fig = plt.figure(figsize=(20,7))
# ax = fig.add_subplot(111)
# plt.plot(range(list_size), np.array([elem[0] for elem in fm_wrist_list]), label='fm_fx')
# plt.plot(range(list_size), np.array([elem[1] for elem in fm_wrist_list]), label='fm_fy')
# plt.plot(range(list_size), np.array([elem[2] for elem in fm_wrist_list]), label='fm_fz')
# plt.plot(range(list_size), np.array([elem[3] for elem in fm_wrist_list]), label='fm_mx')
# plt.plot(range(list_size), np.array([elem[4] for elem in fm_wrist_list]), label='fm_my')
# plt.plot(range(list_size), np.array([elem[5] for elem in fm_wrist_list]), label='fm_mz')
# plt.title("measured force/torque wrist")
# plt.legend()
# plt.show()