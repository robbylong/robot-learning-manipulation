
from gym.envs.registration import register
import gym
from gym.spaces.box import Box
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

from rpl_pybullet_sample_env.pybullet_robots.arms.panda import RPL_Panda, PyBullet_Panda
from rpl_pybullet_sample_env.utils import distance, getBaseTransformMatrix, getModifiedTransformMatrix, getBaseTransformMatrixBatch
import pdb

register(
    id='PandaReacherLimitZ-v0', 
    entry_point='rpl_pybullet_sample_env.pybullet_rl_envs.franka_gym_env:FrankaPandaEnv_LimitZ_v0'
)
register(
    id='PandaReacherLimitZY-v0', 
    entry_point='rpl_pybullet_sample_env.pybullet_rl_envs.franka_gym_env:FrankaPandaEnv_LimitZY_v0'
)


class FrankaPandaEnv(gym.Env):
    """An AI gym environment with the franka emika Panda robot"""

    metadata = {'render.modes': ['human']}  
    # timestep = 1 KHz not 0.1 s
    def __init__(self, use_ee = True, controlMode = 'Delta', simMode = 'direct', 
                urdf_path = '', controlStep=1/24., delta_limits=0.01):
        self.controlModes = {
            'position': p.POSITION_CONTROL, 
            'Delta': p.POSITION_CONTROL, 
            'velocity': p.VELOCITY_CONTROL,
            'torque': p.TORQUE_CONTROL}
        self.simModes = {'gui': p.GUI,'direct': p.DIRECT}
        assert simMode in self.simModes.keys(), f"Simulation mode {simMode} not in {self.simModes.keys()}"
        assert controlMode in self.controlModes.keys(), f"Control mode {controlMode} not in {self.controlModes.keys()}"
        self.simMode = self.simModes[simMode]
        self.controlMode = self.controlModes[controlMode]
        self.controlModeName = controlMode

        if use_ee:
            self.DOF = 9
        else:
            self.DOF = 7

        # start pybullet client
        self.clientId = p.connect(self.simMode)

        self.timeElasped = 0.0

        #TODO: Set simulation step here (for Euler physics integration).
        self.time_step = 1/240.
        p.setTimeStep(self.time_step, physicsClientId=self.clientId) # default 240 Hz = 1/240.

        # downSampling time (number of time steps to apply the same action for)
        self.skipSteps = int(controlStep/self.time_step)

        # load ground plane
        # and gravity pointing down at Z-axis
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setGravity(0,0,-9.81, physicsClientId=self.clientId)

        table_scale = 2.0

        # load table below the robot (table has height 0.625)
        self.tableId = p.loadURDF("table/table.urdf",basePosition=[0.,0.0,-0.625*table_scale], physicsClientId=self.clientId, globalScaling=table_scale)

        # load ground plane under the table
        self.planeId = p.loadURDF("plane.urdf", basePosition=[0.0,0.0,-0.625*table_scale], physicsClientId=self.clientId)

        # load franka_panda robot from rpl_pybulletenv.
        self.BaseOrientation = p.getQuaternionFromEuler([0,0,0])
        self.BasePos = [0,0,0]
        self.initJointPositions=[0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.785, 0.02, 0.02]

        # TODO: allow for different URDF

        # for self-collision checking lib
        root_path = os.path.dirname(__file__)+"/"
        self.rpl_panda = RPL_Panda(self.BasePos, self.BaseOrientation, 
                            init_joints = self.initJointPositions,
                            sim_camera=False, 
                            physicsClient=self.clientId,
                            use_ghost=False,
                            root_path=root_path)
        
        # specify state and action spaces from joint limit info.
        # joint position limits
        if use_ee:
            # action space
            if self.controlModeName == "Delta":
                self.action_space = Box(
                    low = np.array([-delta_limits]* self.DOF),
                    high= np.array([delta_limits]* self.DOF)
                )
            elif self.controlMode is p.POSITION_CONTROL:
                self.action_space = Box(
                    low = np.array(self.rpl_panda.arm_ll+self.rpl_panda.gripper_ll),
                    high= np.array(self.rpl_panda.arm_ul+self.rpl_panda.gripper_ul)
                )
            elif self.controlMode is p.VELOCITY_CONTROL:
                self.action_space = Box(
                    low = np.zeros(self.DOF, dtype=np.float32),
                    high= np.array(self.rpl_panda.arm_vl+self.rpl_panda.gripper_vl)
                )            
            else:
                self.action_space = Box(
                    low = np.zeros(self.DOF, dtype=np.float32),
                    high= np.array(self.rpl_panda.arm_el+self.rpl_panda.gripper_el)
                )
            
            # observation space
            self.observation_space = Box(
                    low = np.array(self.rpl_panda.arm_ll+self.rpl_panda.gripper_ll),
                    high= np.array(self.rpl_panda.arm_ul+self.rpl_panda.gripper_ul)
                )
        else: 
            # use 7 DOF arm (frozen gripper joint)
            if self.controlModeName == "Delta":
                self.action_space = Box(
                    low = np.array([-delta_limits]* self.DOF),
                    high= np.array([delta_limits]* self.DOF)
                )
            elif self.controlMode is p.POSITION_CONTROL:
                self.action_space = Box(
                    low = np.array(self.rpl_panda.arm_ll),
                    high= np.array(self.rpl_panda.arm_ul)
                )
            elif self.controlMode is p.VELOCITY_CONTROL:
                self.action_space = Box(
                    low = np.zeros(self.DOF, dtype=np.float32),
                    high= np.array(self.rpl_panda.arm_vl)
                )            
            else:
                self.action_space = Box(
                    low = np.zeros(self.DOF, dtype=np.float32),
                    high= np.array(self.rpl_panda.arm_el)
                )
            
            # observation space
            self.observation_space = Box(
                    low = np.array(self.rpl_panda.arm_ll),
                    high= np.array(self.rpl_panda.arm_ul)
                )
        
        # Task Info
        self.target = np.array([-0.4,0.0,0.3])
        self.target_radius = 0.04
        visual_kwargs = {
            "radius": self.target_radius,
            "specularColor": [0, 0, 0],
            "rgbaColor": [0.168, 0.870, 0.278, 0.5],
        }
        baseCollisionShapeIndex = -1
        baseVisualShapeIndex = p.createVisualShape(p.GEOM_SPHERE, **visual_kwargs)
        p.createMultiBody(
        baseVisualShapeIndex=baseVisualShapeIndex,
        baseCollisionShapeIndex=baseCollisionShapeIndex,
        baseMass=0.0,
        basePosition=self.target,
        physicsClientId=self.clientId
        )

        # Modified DH matrix
        self.DH = np.array([
            [0,  0.333,  0,           ],   #r_shoulder_pan_joint
            [0,  0,     -1.5708,      ],   # r_shoulder_lift_joint
            [0,  0.316,     1.5708,   ],      # r_upper_arm_joint
            [0.0825,  0,      1.5708, ],        # r_elbow_flex_joint
            [-0.0825,  0.384, -1.5708,],         # r_forearm_roll_joint
            [0,  0,      1.5708,      ],   # r_wrist_flex_joint
            [0.088,  0,        1.5708,],          # r_wrist_roll_joint
            [0,  0.107,        0      ]  # flange
        ], dtype=np.float64)

    def compute_forward_kinematics(self, joint_states, base_positions=None):
        """
        Calculates forward kinematics for batch of joint states (shape: B x D)
        B - batch size
        D - Dimensionality or DOF

        returns batch of XYZ positions for the end effector.
        """
        if len(joint_states.shape) == 1:
            joint_states = joint_states[np.newaxis,:]
        
        B, D = joint_states.shape

        batch_ones = np.ones(B)
        batch_zeros = np.zeros(B)
        
        if base_positions is None:
            base_transform_mat = getBaseTransformMatrix(self.BasePos)
            H_transform = np.repeat(base_transform_mat, B, axis=0)
        else:
            H_transform = getBaseTransformMatrixBatch(base_positions, batch_zeros, batch_ones)

        for i in range(7):
            Theta_Transform = getModifiedTransformMatrix(joint_states[:,i], self.DH[i,0], self.DH[i,1], self.DH[i,2], batch_zeros, batch_ones)
            H_transform = np.matmul(H_transform, Theta_Transform)

        # last transformation for flange (thetas = batch zeros)
        Theta_Transform = getModifiedTransformMatrix(batch_zeros, self.DH[-1,0], self.DH[-1,1], self.DH[-1,2], batch_zeros, batch_ones)
        H_transform = np.matmul(H_transform, Theta_Transform)
        
        return H_transform[:,0:3,-1]

        
    def get_ee_pose_pybullet(self, debug=False):
        return self.rpl_panda.get_ee_pose()
    
    def get_ee_pose_from_joints(self, joints):
        pose = self.rpl_panda.get_ghost_ee_pose(joints)
        return np.array(pose[0])


    def reset(self, state=None):

        if state is None:
            state = np.array(self.initJointPositions[:self.DOF], "float32")

        if not isinstance(state, np.ndarray):
            state = np.array(state, "float32")

        if len(state) > self.DOF:
            self.rpl_panda.reset_joints(state[:self.DOF])
            velocity = np.zeros(self.DOF)
            self.obs = np.hstack([state[:self.DOF], velocity])
        else:
            velocity = np.zeros_like(state)
            self.rpl_panda.reset_joints(state)
            self.obs = np.hstack([state, velocity])
        
        return self.obs

    def get_observation(self):
        """
        After having called env.reset(), we use this function to read the state of the robot from pybullet
        returnForceApplied - returns the action/control that was applied by the simulator at before stepSimulation.
        """

        #TODO: overwrite this function for different tasks to include the state of task objects.
        joint_state, joint_velocity = self.rpl_panda.get_joint_state(DOF=self.DOF)
        return np.hstack([joint_state, joint_velocity])


    def render(self):
        # TODO: Implement a render function which shows the robot from the view of an overhead camera 
        # similar to the Hindsight Experience Replay paper
        pass

    def close(self):
        p.disconnect(self.clientId)   
        
    def seed(self, seed=None): 
        """Returns a seeded random number generator (i.e., self.np_random) and the seed """
        self.np_random, seed = gym.utils.seeding.np_random(seed)    
        return [seed]
        
    def step(self, action):
        """Applies the same action for skipSteps times then returns next_obs and rewards for the final state."""
        if self.controlModeName == "Delta":
            target = action + self.obs[:self.DOF]
            ll = (self.rpl_panda.arm_ll+self.rpl_panda.gripper_ll)[:self.DOF]
            ul = (self.rpl_panda.arm_ul+self.rpl_panda.gripper_ul)[:self.DOF]
            target = np.clip(target, ll, ul)
        else: 
            target = action 

        # apply action
        for i in range(self.skipSteps): #sim_step*robot_step 
            self.rpl_panda.apply_joint_control(controls=target, DOF=self.DOF, mode=self.controlMode)
            p.stepSimulation(physicsClientId=self.clientId)

        self.timeElasped += self.skipSteps*self.time_step
        
        new_obs = self.get_observation()

        self.obs = new_obs 
        # compute reward (depends only on the observation)
        r = self.compute_reward(new_obs)
        # print(self.timeElasped)
        # to keep trajectory sampling simple, keep all episodes the same length (no early termination)
        done = False
        # if self.timeElasped > 1.0:
        #     # pdb.set_trace()
        #     p.changeVisualShape(objectUniqueId=self.collision_plane_body , linkIndex=self.collision_plane_id, rgbaColor=[1.0,0.0,0.0,0.3],physicsClientId=self.clientId)

        return new_obs,r, done, {}

    def compute_reward(self,obs):
        """overwrite this functions for different environments."""
        raise NotImplementedError("The base class reward function needs to be overwritten")

class FrankaPandaEnv_WallConstraintReacher(FrankaPandaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def make_collision_plane(self, half_extents, position, rgba_color=[0.0,0.0,1.0,0.3]):
        collision_plane_id = p.createVisualShape(shapeType=p.GEOM_BOX,  
                                                      halfExtents = half_extents, 
                                                      rgbaColor=rgba_color)
        collision_plane_body = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=collision_plane_id,
            basePosition=position
            )
        return 
    def compute_reward(self,obs):
        """overwrite this functions for different environments."""
        curr_pos, curr_ori = self.get_ee_pose_pybullet()
        d = distance(self.target, np.array(curr_pos))
        # In the default env, we perform a reacher task
        return np.exp(-d)


class FrankaPandaEnv_LimitZ_v0(FrankaPandaEnv_WallConstraintReacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # objectUid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.7,0,0.1])
        self.Z_constraint = 0.85
        # add collision plane
        # place this in a separate env file later
        self.make_collision_plane(half_extents=[1.0, 1.0, 0.005], 
                                  position=[0,0,self.Z_constraint])

    def constraint_function(self, batch_obs, batch_actions=None):
        """
        Checks for constraint violation in batch of.
        """
        if len(batch_obs.shape) == 1:
            batch_obs = batch_obs[np.newaxis,:]
        
        B, D = batch_obs.shape

        # slice away velocities
        batch_joint_states = batch_obs[:,:self.DOF]

        # compute fk
        batch_eef_pos = self.compute_forward_kinematics(batch_joint_states) # B x 3

        # get z values
        batch_eef_pos_z = batch_eef_pos[:,2] # (B,)

        # check for violations
        return np.where(batch_eef_pos_z > self.Z_constraint, 1.0, 0.0)



class FrankaPandaEnv_LimitZY_v0(FrankaPandaEnv_WallConstraintReacher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # objectUid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.7,0,0.1])
        self.Z_constraint   = 0.85
        self.Y_constraint_L = 0.4
        self.Y_constraint_R =-self.Y_constraint_L
        # add collision plane
        # place this in a separate env file later
        self.make_collision_plane(half_extents=[1.0, self.Y_constraint_L, 0.005], 
                                  position=[0,0,self.Z_constraint])
        self.make_collision_plane(half_extents=[1.0, 0.005, self.Z_constraint], 
                                  position=[0, self.Y_constraint_L,0])
        self.make_collision_plane(half_extents=[1.0, 0.005, self.Z_constraint], 
                                  position=[0, self.Y_constraint_R,0])

    def constraint_function(self, batch_obs, batch_actions=None):
        """
        Checks for constraint violation in batch of.
        """
        if len(batch_obs.shape) == 1:
            batch_obs = batch_obs[np.newaxis,:]
        
        B, D = batch_obs.shape

        # slice away velocities
        batch_joint_states = batch_obs[:,:self.DOF]

        # compute fk
        batch_eef_pos = self.compute_forward_kinematics(batch_joint_states) # B x 3

        # get z values
        batch_eef_pos_x = batch_eef_pos[:,0] # (B,)
        batch_eef_pos_y = batch_eef_pos[:,1] # (B,)
        batch_eef_pos_z = batch_eef_pos[:,2] # (B,)

        # check for violations
        violation_z = np.where(batch_eef_pos_z > self.Z_constraint, 1.0, 0.0)
        violation_y_l = np.where(batch_eef_pos_y > self.Y_constraint_L, 1.0, 0.0)
        violation_y_r = np.where(batch_eef_pos_y < self.Y_constraint_R, 1.0, 0.0)
        
        violations = np.logical_or(violation_z,violation_y_l, violation_y_r)

        return violations


    