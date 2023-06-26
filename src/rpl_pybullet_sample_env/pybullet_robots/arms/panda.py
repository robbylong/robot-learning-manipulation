import pybullet as p
from rpl_pybullet_sample_env.pybullet_robots.robot  import Robot
from rpl_pybullet_sample_env.pybullet_robots.arms.collision_utils import CollisionDetector
import math
import numpy as np

# PyKin
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.collision.collision_manager import CollisionManager

# load robot models
import robot_models
import os
import inspect

MODEL_PATH = os.path.dirname(inspect.getfile(robot_models))

PATH_TO_RPL_PANDA = MODEL_PATH+\
                    "/franka_description/robots/panda_arm_hand_rs.urdf.urdf"
PATH_TO_RPL_PANDA_WITH_LUKE_GRIPPER = MODEL_PATH+\
                    "/gripper_description/urdf/finger5/panda_and_gripper_N5_1000i.urdf"
                    #"/home/dennisushi/repos/temp/gripper_description/urdf/"+\
PATH_TO_RPL_PANDA_COLLISION =  MODEL_PATH+\
                    "/franka_description/robots/panda_arm_hand_rs_collisions.urdf.urdf"

assert os.path.isfile(PATH_TO_RPL_PANDA), "Panda URDF not found - check if install was correct"

print(PATH_TO_RPL_PANDA)
# PyBullet Panda
class PandaRobot(Robot):
    def __init__(self, path, pos, ori, init_joints, root_path = None, use_pykin=True,
                    use_real_force=False):
        """ Simulates URDF from given path at given
        # position/orientation
        # path : string : relative or absolute path to URDF/Xacro
        #                 note, PyBullet needs reference-less URDF
        # pos : [x,y,z] list-like position
        # ori : [x,y,z,w] list-like quaternion 
        # init_joints : initial joint positions for Panda arm joints
        """

        super().__init__(path, pos, ori, useFixedBase=True,
                                        flags = p.URDF_USE_SELF_COLLISION)
        self.home_joint_pos = init_joints
        self.initialize_joints(init_joints)
        self.arm_dof = 7 # Arm degrees of freedom
        self.ee_dof  = 2 # gripper degrees of freedom

        #################################
        # Limits used for motor control #
        #################################
        # Joint position limits, taken from Franka Emika Panda official URDF
        self.arm_ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.arm_ul = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
        self.gripper_ll = [0.00, 0.00]
        self.gripper_ul = [0.04, 0.04]
        # Joint velocity limits, taken from Franka Emika Panda official URDF
        self.arm_vl = [2.1750,2.1750,2.1750,2.1750,2.61,2.61,2.61]
        self.gripper_vl = [0.2,0.2]
        # Joint effort limits, taken from Franka Emika Panda official URDF
        self.arm_el = [87,87,87,87,12,12,12]
        self.gripper_el = [20,20]

        self.max_velocity = 1.0
        
        # TODO: update joint range with real value, taken from PyBullet
        self.joint_range = [7]*self.arm_dof 

        # create Pykin robot with URDF collision model
        euler_ori = p.getEulerFromQuaternion(ori)
        self.use_pykin = use_pykin
        if use_pykin:
            self.pykin_robot = SingleArm(PATH_TO_RPL_PANDA_COLLISION, Transform(rot=euler_ori, pos=pos))
            # create Pykin Collsion Manager
            if root_path:
                ROOT_PATH =  root_path
            else:
                ROOT_PATH = ""
            self.pykin_col_manager = CollisionManager(ROOT_PATH)
        
        self.use_real_force = use_real_force
        if not use_real_force:
            print("INFO: The Panda motor control function uses infinite force."\
                "To use real force, set use_real_force flag to True")
        return
    
    def check_self_collisions(self, joints):
        """
        Checks for pairwise collisions between the links of the robot. This function uses the 
        PyKin library and requires the urdf to only use stl meshes for collision
        """
        method = "PyKin"
        if method == "PyKin":
          # first get FK transforms
          fk_dict = self.pykin_robot.forward_kin(joints)
          self.pykin_col_manager.setup_robot_collision(self.pykin_robot, fk_dict)
          # Check self Collision 
          # TODO: Improve function using moveit!'s allowed collision matrix 
          result, name, data = self.pykin_col_manager.in_collision_internal(return_names=True, return_data=True)
          print(result, name)
        else:
          pass
        return result
    
    def compute_fk(self, joints):
        fk_dict = self.pykin_robot.forward_kin(joints)
        ee_link_transform = fk_dict['panda_link8']
        displacement = [0,0,0.113]
        pos_n, ori_n = p.multiplyTransforms(ee_link_transform.pos, ee_link_transform.rot, 
                                    displacement, 
                                    [0,0,0,1])
        return pos_n, ori_n

    def apply_joint_control(self, controls, DOF, mode=p.POSITION_CONTROL):
        el = self.arm_el+self.gripper_el
        if mode != p.POSITION_CONTROL:
            raise NotImplementedError("This feature isn't implemented yet")
        for i in range(DOF): # the first 7 joints correspond to the arm joints
            if not self.use_real_force: 
                p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                    p.POSITION_CONTROL, controls[i],
                                    #force=el[i],
                                    maxVelocity=self.max_velocity)            
            else: 
                p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                    p.POSITION_CONTROL, controls[i],
                                    force=el[i],
                                    maxVelocity=self.max_velocity)  
        # TODO: Implement other control modes
        

    def get_joint_state(self, DOF):
        out = p.getJointStates(self.id,list(range(1,DOF+1)))
        JointStates = [elem[0] for elem in out]
        JointVelocities = [elem[1] for elem in out]
        
        return np.array(JointStates, "float32"), np.array(JointVelocities, "float32")

    def initialize_joints(self, initial_joints):
        index = 0
        self.joint_ids = []
        # print("the number of joints is: %d", p.getNumJoints(self.id))
        for j in range (p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.id, j)
            print(info)
            if (info[2] == p.JOINT_PRISMATIC) or (info[2] == p.JOINT_REVOLUTE):
                jpos = initial_joints[index]
                p.resetJointState(self.id, j, jpos) 
                index=index+1
                print ( "Joint %d %s is initialized to %0.2f"%(
                         info[0],info[1],jpos), info[-5] )
                self.joint_ids.append(info[0])
            else:
                print ("Skipping joint %d %s"%(info[0], info[1]), info[-5])
        
        print(index)
        return

    def reset_joints(self, joints):
        for i, joint_val in enumerate(joints):
          p.resetJointState(self.id, i+1, joint_val)
        return

    def example_movement(self, timestep, freq=1./60.):
        t = timestep
        pos = [self.init_pos[0] + 0.3 + 0.2 * math.sin(1.5 * t), 
               self.init_pos[1] + 0.0 + 0.1 * math.cos(1.5 * t), 
               self.init_pos[2] + 0.6]
        orn = p.getQuaternionFromEuler([math.pi,0., 0.])
        jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                  self.arm_ll, self.arm_ul, self.joint_range , 
                                                  self.home_joint_pos, maxNumIterations=5)
        for i in range(self.arm_dof): # the first 7 joints correspond to the arm joints
            p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                p.POSITION_CONTROL, jointPoses[i], 
                                force=5 * 240.)
        pass
    
class PyBullet_Panda(PandaRobot):
    def __init__(self, pos, ori, init_joints, use_pykin=True):
        # Simulates PyBullet Panda at given position/orientation
        # pos : [x,y,z] list-like position
        # ori : [x,y,z,w] list-like quaternion 
        # init_joints : initial joint positions for Panda arm joints
        
        super().__init__("franka_panda/panda.urdf", pos, ori, init_joints, use_pykin)
        self.ee_index = 11

class RPL_Panda(PandaRobot):
    """ RPL Panda class """
    def __init__(self, pos, ori, init_joints, physicsClient=None, 
        sim_camera=True, use_gui=True, use_ghost=True, root_path=None, use_pykin=True,
        path_to_panda_urdf = PATH_TO_RPL_PANDA):
        """Simulates RPL Panda at given position/orientation
        pos : [x,y,z] list-like position
        ori : [x,y,z,w] list-like quaternion
        init_joints : initial joint positions for Panda arm joints
        sim_camera : bool : whether to simulate RGBD camera or not.
        use_gui : bool : whether to simulate in GUI or not
        use_ghost : bool : whether to spawn a ghost robot or not
        """
        super().__init__(path_to_panda_urdf, pos, ori,
                        init_joints, root_path=root_path, use_pykin=use_pykin)
        # ee index is not the default one because of different links
        self.ee_index = 9
        self.sim_camera = sim_camera
        self.use_gui = use_gui
        self.physicsClient = physicsClient
        self.use_ghost = use_ghost
        if self.use_ghost:
            print("Using ghost robot clone")
            self._create_ghost(init_joints, pos, ori)

        if sim_camera and use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Set up Collision Detector
        self.virtual_links = [9,12] # links that dont have a physical link.
        # joint 9 is from panda link 8 to panda_hand (45 degree rotation)
        # joint 12 is for the virtual camera link.
        self.pybul_col_manager = CollisionDetector(self.physicsClient, self, use_ghost=self.use_ghost,
                                                    virtual_links=self.virtual_links)

    def check_self_collisions(self, joints):
        """
        Checks for pairwise collisions between the links of the robot. This function uses the 
        PyKin library and requires the urdf to only use stl meshes for collision
        """
        method = "PyBullet"#"PyKin"
        if method is None: return True
        if method == "PyKin":
          # first get FK transforms
          fk_dict = self.pykin_robot.forward_kin(joints)
          self.pykin_col_manager.setup_robot_collision(self.pykin_robot, fk_dict)
          # Check self Collision 
          # TODO: Improve function using moveit!'s allowed collision matrix 
          result, name, data = self.pykin_col_manager.in_collision_internal(return_names=True, return_data=True)
          print(result, name)
        elif "PyBullet":
          result = self.pybul_col_manager.in_collision(joints)
          print("Collisison",result)
        else: raise NotImplementedError("PyKin or PyBullet available only")
        return result

    def _create_ghost(self, init_joints, pos, ori):
        """ 
        Creates a ghost robot. For Forward kinematics. We only need this because we don't want to move the robot to a 
        joint config before computing the forward kinematics. So we create a 2nd 'ghost robot' which we can move into 
        different configurations.
        """
        ghost_pos = [pos[0], pos[1], pos[2]]
        ghost_ori = [ori[0],ori[1],ori[2],ori[3]]
        self.ghost = PandaRobot(PATH_TO_RPL_PANDA, ghost_pos, ghost_ori, init_joints, use_pykin=False)
        n_jnts = p.getNumJoints(self.ghost.id, physicsClientId=self.physicsClient)
        for link_index in range(n_jnts):
          p.changeVisualShape(self.ghost.id, link_index, rgbaColor=[1,0.5,0,0.3])
          p.setCollisionFilterGroupMask(self.ghost.id, link_index, 0, 0)
        return

    def get_ee_pose(self, robot_id=None, debug=None):
        """ Function to get the EE pose of the Panda
        :param physicsClient: PyBullet physics client
        :param debug: debug, optional, if set - it must be the
                      object id of the visualization object that
                      will be placed at the EE point.
        """
        # Displamcent from panda_hand/panda_link8 to
        # the end effector.
        displacement = [0,0,0.113]
        # Panda link 8 is link 8, if you want the
        # orientation of Panda_hand, use 9 instead
        link_id = self.ee_index
        if robot_id is None:
            ls = p.getLinkState(self.id, link_id,
                                physicsClientId=self.physicsClient)
        else:
            ls = p.getLinkState(robot_id, link_id,
                                physicsClientId=self.physicsClient)

        # The position and orientation of the parent link
        pos = ls[0]
        ori = ls[1]
        # the position and orientation of the EE
        pos_n, ori_n = p.multiplyTransforms(pos, ori, 
                                            displacement, 
                                            [0,0,0,1])
        # visualization
        if debug is not None:
          p.resetBasePositionAndOrientation(debug, pos_n, ori_n,
                            physicsClientId=self.physicsClient)
        return pos_n, ori_n

    def get_ghost_ee_pose(self, joints, debug=None):
        """
        Returns the EE pose given joints by computing FK
        on the ghost clone
        : param joints: List, list of joint values
        : physicsClient: int, Pybullet Physics Client
        """
        if not self.use_ghost: 
          raise ValueError("Set use_ghost to true if using ghost EE functions")
        for i, joint_val in enumerate(joints):
          p.resetJointState(self.ghost.id, i+1, joint_val)
        pos, ori = self.get_ee_pose(robot_id=self.ghost.id, debug=debug)
        return pos, ori

    def synth_camera(self):
        """Synthesize camera"""
        pixelWidth = 640
        pixelHeight = 480
        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100

        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

        li = p.getJointInfo(self.id, 15)[-1] # pareent link index
        com_p, com_o, _, _, _, _ = p.getLinkState(self.id, li, 
                                        computeForwardKinematics=True)

        rot_matrix = p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (1, 0, 0) # z-axis
        init_up_vector = (0, 1, 0) # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        img = p.getCameraImage(pixelWidth, pixelHeight, view_matrix, projection_matrix)
        self.last_rgb_msg = img [2]
        self.last_depth_msg = img [3]
        self.last_segm_msg = img [4]

        return

    def step (self):
        """Step simulation"""
        if self.sim_camera:
            self.synth_camera()
        return
    

class PandaWithCustomGripper(RPL_Panda):
    """ RPL Panda class """
    def __init__(self, pos, ori, init_joints, physicsClient=None, 
        sim_camera=True, use_gui=True, use_ghost=False, root_path=None, use_pykin=False,
        path_to_panda_urdf = PATH_TO_RPL_PANDA_WITH_LUKE_GRIPPER):
        """Simulates RPL Panda at given position/orientation
        pos : [x,y,z] list-like position
        ori : [x,y,z,w] list-like quaternion
        init_joints : initial joint positions for Panda arm joints
        sim_camera : bool : whether to simulate RGBD camera or not.
        use_gui : bool : whether to simulate in GUI or not
        use_ghost : bool : whether to spawn a ghost robot or not
        """
        super().__init__(pos=pos, ori=ori, init_joints=init_joints, 
        physicsClient=physicsClient, sim_camera=sim_camera, use_gui=use_gui, 
        use_ghost=use_ghost, root_path=root_path, use_pykin=use_pykin,
        path_to_panda_urdf = path_to_panda_urdf)

        # super().__init__(path_to_panda_urdf, pos, ori,
        #                 init_joints, root_path=root_path, use_pykin=use_pykin)
        # ee index is not the default one because of different links
        self.ee_index = 9
        self.sim_camera = sim_camera
        self.use_gui = use_gui
        self.physicsClient = physicsClient
        self.use_ghost = use_ghost

        # if len(init_joints) != 14:
        #     raise NotImplementedError("RPL Panda with Custom Luke Gripper"\
        #         "currently supports only 14 joints (1-segment), got %d"%len(init_joints))

        if self.use_ghost:
            print("Using ghost robot clone")
            self._create_ghost(init_joints, pos, ori)

        if sim_camera and use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            # p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Set up Collision Detector
        self.virtual_links = [9,12] # links that dont have a physical link.
        # joint 9 is from panda link 8 to panda_hand (45 degree rotation)
        # joint 12 is for the virtual camera link.
        self.pybul_col_manager = CollisionDetector(self.physicsClient, self, use_ghost=self.use_ghost,
                                                    virtual_links=self.virtual_links)

        # Arm degrees of freedom
        self.arm_dof = 7
        self.ee_dof  = 7 # gripper degrees of freedom,  with no segments: 7 

        #################################
        # Limits used for motor control #
        #################################
        # self.gripper_ll = [49e-3, -0.7, 49e-3, -0.7, 49e-3, -0.7, 0.0e-3]
        ROT_LIMIT = 0.7
        self.gripper_ll = [49e-3, -ROT_LIMIT, 49e-3, -ROT_LIMIT, 49e-3, -ROT_LIMIT, 0.0e-3]
        self.gripper_ul = [134e-3, ROT_LIMIT, 134e-3, ROT_LIMIT, 134e-3, ROT_LIMIT, 165e-3]
        self.gripper_vl = [20e-3, 0.2, 20e-3, 0.2, 20e-3, 0.2, 20e-3]
        self.gripper_el = [20] * 7
        self.grip_rot = 0
        self.grip_pris = 0
        self.grip_palm = 0
        self.base_height = 0

    def example_movement(self, timestep, freq=1./60.):
        t = timestep
        pos = [self.init_pos[0] + 0.35 + 0.1 * math.sin(1.5 * t), 
               self.init_pos[1] + 0.0 + 0.1 * math.cos(1.5 * t), 
               self.init_pos[2] + 0.7]
        orn = p.getQuaternionFromEuler([math.pi,0., 0.])
        jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                  self.arm_ll, self.arm_ul, self.joint_range , 
                                                  self.home_joint_pos, maxNumIterations=5)
        jointPoses = np.asarray(jointPoses)

        grip_rot_range = 0.4
        grip_rot_mid =  0 
        grip_rot = grip_rot_mid + (grip_rot_range/2) * math.sin(1.5 * t + 3.14)


        grip_pris_range = (self.gripper_ul[0]-self.gripper_ll[0])
        grip_pris_mid =  self.gripper_ll[0] + grip_pris_range/2
        grip_pris = grip_pris_mid + (grip_pris_range/2) * math.sin(1.5 * t)

        grip_palm_range = (self.gripper_ul[-1]-self.gripper_ll[-1])
        grip_palm_mid =  self.gripper_ll[-1] + grip_palm_range/2
        grip_palm = grip_palm_mid + (grip_palm_range/2) * math.sin(1.5 * t)

        jointPoses[self.arm_dof+1] = grip_rot
        jointPoses[self.arm_dof+3] = grip_rot
        jointPoses[self.arm_dof+5] = grip_rot

        jointPoses[self.arm_dof] = grip_pris
        jointPoses[self.arm_dof+2] = grip_pris
        jointPoses[self.arm_dof+4] = grip_pris

        jointPoses[self.arm_dof+6] = grip_palm
        
        for i in range(self.arm_dof+self.ee_dof): # the first 7 joints correspond to the arm joints
            p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                p.POSITION_CONTROL, jointPoses[i], 
                                force=5 * 240.)
        pass

    def move_upward(self, timestep, freq=1./60.):
        t = timestep

        pos = [self.init_pos[0] + 0.4 - 0.4*t , 
        self.init_pos[1] + 0.4*t, 
        self.init_pos[2] + 0.7]
        orn = p.getQuaternionFromEuler([0,0,0])
        jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                            self.arm_ll, self.arm_ul, self.joint_range , 
                                            self.home_joint_pos, maxNumIterations=5)
        jointPoses = np.asarray(jointPoses)
        for i in range(self.arm_dof+self.ee_dof): # the first 7 joints correspond to the arm joints
            p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                p.POSITION_CONTROL, jointPoses[i], 
                                force=5 * 240.)
        pass

    def get_base_state(self, robot_id=None, debug=None):
        """ Function to get the EE pose of the Panda
        position is calibrated such that position_z=0 means the gripper finger is 10mm above the ground 
        :param physicsClient: PyBullet physics client
        :param debug: debug, optional, if set - it must be the
                      object id of the visualization object that
                      will be placed at the EE point.
        """
        # Displamcent from panda_hand/panda_link8 to
        # the end effector.
        displacement = [0,0,0.290]
        # Panda link 8 is link 8, if you want the
        # orientation of Panda_hand, use 9 instead
        link_id = self.ee_index
        if robot_id is None:
            ls = p.getLinkState(self.id, link_id,
                                physicsClientId=self.physicsClient)
        else:
            ls = p.getLinkState(robot_id, link_id,
                                physicsClientId=self.physicsClient)

        # The position and orientation of the parent link
        pos = ls[0]
        ori = ls[1]
        # the position and orientation of the EE
        pos_n, ori_n = p.multiplyTransforms(pos, ori, 
                                            displacement, 
                                            [0,0,0,1])
        # visualization
        if debug is not None:
          p.resetBasePositionAndOrientation(debug, pos_n, ori_n,
                            physicsClientId=self.physicsClient)
        
        # calibrate the position
        pos_n = list(pos_n)  # Convert the tuple to a list
        pos_n[2] -= (1.2492 + 0.0078825 + 0.01088)
        return pos_n[2]
    
    def move_to_above(self, timestep, obj_x,obj_y, obj_z):
        t = timestep

        pos = [self.init_pos[0] + (obj_x-self.init_pos[0])*t, 
        self.init_pos[1] + 0.4 + (obj_y-(self.init_pos[1]+0.4))*t, 
        self.init_pos[2] + 0.7 + (obj_z + 0.300 -(self.init_pos[2]+0.7))*t]
        orn = p.getQuaternionFromEuler([0,0,0])
        jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                            self.arm_ll, self.arm_ul, self.joint_range , 
                                            self.home_joint_pos, maxNumIterations=5)
        jointPoses = np.asarray(jointPoses)
        for i in range(self.arm_dof+self.ee_dof): # the first 7 joints correspond to the arm joints
            p.setJointMotorControl2(self.id, self.joint_ids[i], 
                                p.POSITION_CONTROL, jointPoses[i], 
                                force=5 * 240.)
        
        self.grip_pris = p.getJointState(self.id, 10)[0]
        self.grip_rot = p.getJointState(self.id, 11)[0]
        self.grip_palm = p.getJointState(self.id, 34)[0]
        
        pass

    def grasp_noRL(self, timestep, obj_x,obj_y, obj_z):
        t = timestep
        pos = [self.init_pos[0] + (obj_x-self.init_pos[0]), 
        self.init_pos[1] + 0.4 + (obj_y-(self.init_pos[1]+0.4)), 
        self.init_pos[2] + 0.7 + (obj_z + 0.3 -(self.init_pos[2]+0.7))]
        orn = p.getQuaternionFromEuler([0, 0., 0.])
        jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                  self.arm_ll, self.arm_ul, self.joint_range , 
                                                  self.home_joint_pos, maxNumIterations=5)
        jointPoses = np.asarray(jointPoses)
        

        grip_rot_range = 0.4
        grip_rot_mid =  0
        grip_rot = grip_rot_mid - (grip_rot_range*6) * 1.5*t 
        # grip_rot = grip_rot_mid 


        grip_pris_range = (self.gripper_ul[0]-self.gripper_ll[0])
        grip_pris_mid =  self.gripper_ll[0] + grip_pris_range/2
        grip_pris = grip_pris_mid + (grip_pris_range*0) * t

        grip_palm_range = (self.gripper_ul[-1]-self.gripper_ll[-1])
        grip_palm_mid =  self.gripper_ll[-1] + grip_palm_range/2
        grip_palm = grip_palm_mid - (grip_palm_range/3) * t

        jointPoses[self.arm_dof+1] = grip_rot
        jointPoses[self.arm_dof+3] = grip_rot
        jointPoses[self.arm_dof+5] = grip_rot

        jointPoses[self.arm_dof] = grip_pris
        jointPoses[self.arm_dof+2] = grip_pris
        jointPoses[self.arm_dof+4] = grip_pris

        jointPoses[self.arm_dof+6] = grip_palm
        # joint_id = [1,2,3,4,5,6,7] + [10,11,18,19,26,27,34]
        joint_id = [1,2,3,4,5,6,7] + [10,11,12,13,14,15,16]
        print(jointPoses)
        for i in range(self.arm_dof+7): # the first 7 joints correspond to the arm joints
            p.setJointMotorControl2(self.id, joint_id[i], 
                                p.POSITION_CONTROL, jointPoses[i], 
                                force=5 * 240.)
    #     pass
    def grasp(self, policy_t,action,obj_x,obj_y,obj_z):
        orn = p.getQuaternionFromEuler([0, 0., 0.])
        # print(policy_t)
        if action == "X_close" or action == "X_open":
            pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height]
            # prismatic joint -= 1mm
            jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                    self.arm_ll, self.arm_ul, self.joint_range , 
                                                    self.home_joint_pos, maxNumIterations=5)
            # for j in range (p.getNumJoints(self.id)):
            #     # p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            #     info = p.getJointInfo(self.id, j)
            #     print(info)
            jointPoses = np.asarray(jointPoses)
            # print(jointPoses)

            grip_rot = self.grip_rot 

            if action == "X_open":
                # if self.grip_pris >= self.gripper_ul[0]:
                #     grip_pris = self.grip_pris
                # else:
                    grip_pris = self.grip_pris + 1e-3*policy_t
            elif action == "X_close":
                # if self.grip_pris <= self.gripper_ll[0]:
                #     grip_pris = self.grip_pris
                # else:
                    grip_pris = self.grip_pris - 1e-3*policy_t
            if round(policy_t,6) == 1:
                print("finished!!!")
                self.grip_pris = grip_pris

            grip_palm = self.grip_palm

            jointPoses[self.arm_dof+1] = grip_rot
            jointPoses[self.arm_dof+8] = grip_rot
            jointPoses[self.arm_dof+15] = grip_rot

            jointPoses[self.arm_dof] = grip_pris
            jointPoses[self.arm_dof+7] = grip_pris
            jointPoses[self.arm_dof+14] = grip_pris

            jointPoses[self.arm_dof+21] = grip_palm
            joint_id = [1,2,3,4,5,6,7] + [10,11,18,19,26,27,34]
            for i in range(self.arm_dof): # the first 7 joints correspond to the arm joints
                p.setJointMotorControl2(self.id, joint_id[i], 
                                    p.POSITION_CONTROL, jointPoses[i], 
                                    force=5*240.)
            p.setJointMotorControl2(self.id, 10, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 11, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 18, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 19, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 26, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 27, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 34, p.POSITION_CONTROL, grip_palm, force=50)
            pass
        
        if action == "Y_close" or action == "Y_open":
            pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height]
            # prismatic joint -= 1mm
            jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                    self.arm_ll, self.arm_ul, self.joint_range , 
                                                    self.home_joint_pos, maxNumIterations=5)
            jointPoses = np.asarray(jointPoses)

            grip_pris = self.grip_pris 

            if action == "Y_open":
                # if self.grip_rot >= self.gripper_ul[1]:
                # # if self.grip_rot >= 0.4:
                #     grip_rot = self.grip_rot
                # else:
                    grip_rot = self.grip_rot + 0.01*policy_t
            elif action == "Y_close":
                # if self.grip_rot <= self.gripper_ll[1]:
                # # if self.grip_rot <= -0.4:
                #     grip_rot = self.grip_rot
                # else:
                    grip_rot = self.grip_rot - 0.01*policy_t
            if round(policy_t,6) == 1:
                print("finished!!!")
                self.grip_rot = grip_rot

            grip_palm = self.grip_palm

            jointPoses[self.arm_dof+1] = grip_rot
            jointPoses[self.arm_dof+8] = grip_rot
            jointPoses[self.arm_dof+15] = grip_rot

            jointPoses[self.arm_dof] = grip_pris
            jointPoses[self.arm_dof+7] = grip_pris
            jointPoses[self.arm_dof+14] = grip_pris

            jointPoses[self.arm_dof+21] = grip_palm
            joint_id = [1,2,3,4,5,6,7] + [10,11,18,19,26,27,34]
            for i in range(self.arm_dof): # the first 7 joints correspond to the arm joints
                p.setJointMotorControl2(self.id, joint_id[i], 
                                    p.POSITION_CONTROL, jointPoses[i], 
                                    force=5*240.)
            p.setJointMotorControl2(self.id, 10, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 11, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 18, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 19, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 26, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 27, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 34, p.POSITION_CONTROL, grip_palm, force=50)
            pass

        if action == "Z_close" or action == "Z_open":
            pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height]
            # prismatic joint -= 1mm
            jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                    self.arm_ll, self.arm_ul, self.joint_range , 
                                                    self.home_joint_pos, maxNumIterations=5)
            jointPoses = np.asarray(jointPoses)

            grip_pris = self.grip_pris 

            grip_rot = self.grip_rot

            if action == "Z_open":
                # if self.grip_palm <= self.gripper_ll[6]:
                #     grip_palm = self.grip_palm
                # else:
                    grip_palm = self.grip_palm - 2e-3*policy_t
            elif action == "Z_close":
                # if self.grip_palm >= self.gripper_ul[6]:
                #     grip_palm = self.grip_palm
                # else:
                    grip_palm = self.grip_palm + 2e-3*policy_t

            if round(policy_t,6) == 1:
                print("finished!!!")
                self.grip_palm = grip_palm

            jointPoses[self.arm_dof+1] = grip_rot
            jointPoses[self.arm_dof+8] = grip_rot
            jointPoses[self.arm_dof+15] = grip_rot

            jointPoses[self.arm_dof] = grip_pris
            jointPoses[self.arm_dof+7] = grip_pris
            jointPoses[self.arm_dof+14] = grip_pris

            jointPoses[self.arm_dof+21] = grip_palm
            joint_id = [1,2,3,4,5,6,7] + [10,11,18,19,26,27,34]
            for i in range(self.arm_dof): # the first 7 joints correspond to the arm joints
                p.setJointMotorControl2(self.id, joint_id[i], 
                                    p.POSITION_CONTROL, jointPoses[i], 
                                    force=5*240.)
            p.setJointMotorControl2(self.id, 10, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 11, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 18, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 19, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 26, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 27, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 34, p.POSITION_CONTROL, grip_palm, force=50)
            pass

        if action == "H_up" or action == "H_down":
            if action == "H_up":
                # if self.get_base_state() >= 30e-3:
                #     pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height]
                # else:
                pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height + 2e-3*policy_t]
                if round(policy_t,6) == 1:
                    print("finished!!!")
                    self.base_height += 2e-3
            elif action == "H_down":
                # if self.get_base_state() <= -30e-3:
                #     pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height]
                # else:
                pos = [obj_x, obj_y, obj_z + 0.3 + self.base_height - 2e-3*policy_t]
                if round(policy_t,6) == 1:
                    print("finished!!!")
                    self.base_height -= 2e-3
            # prismatic joint -= 1mm
            jointPoses = p.calculateInverseKinematics(self.id, self.ee_index, pos, orn, 
                                                    self.arm_ll, self.arm_ul, self.joint_range , 
                                                    self.home_joint_pos, maxNumIterations=5)

            jointPoses = np.asarray(jointPoses)

            grip_pris = self.grip_pris 

            grip_rot = self.grip_rot

            grip_palm = self.grip_palm

            jointPoses[self.arm_dof+1] = grip_rot
            jointPoses[self.arm_dof+8] = grip_rot
            jointPoses[self.arm_dof+15] = grip_rot

            jointPoses[self.arm_dof] = grip_pris
            jointPoses[self.arm_dof+7] = grip_pris
            jointPoses[self.arm_dof+14] = grip_pris

            jointPoses[self.arm_dof+21] = grip_palm
            joint_id = [1,2,3,4,5,6,7] + [10,11,18,19,26,27,34]
            for i in range(self.arm_dof): # the first 7 joints correspond to the arm joints
                p.setJointMotorControl2(self.id, joint_id[i], 
                                    p.POSITION_CONTROL, jointPoses[i], 
                                    force=5*240.)
            p.setJointMotorControl2(self.id, 10, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 11, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 18, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 19, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 26, p.POSITION_CONTROL, grip_pris, force=30)
            p.setJointMotorControl2(self.id, 27, p.POSITION_CONTROL, grip_rot, force=30)
            p.setJointMotorControl2(self.id, 34, p.POSITION_CONTROL, grip_palm, force=50)
            pass

        
