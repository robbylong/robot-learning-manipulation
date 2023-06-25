import pybullet as p
import pybullet_data as pd
from rpl_pybullet_sample_env.pybullet_robots.robot  import Robot
import math
import time
import numpy as np


class PyBullet_LaikaGo(Robot):
    def __init__(self, pos, ori):
        super().__init__("laikago/laikago_toes.urdf", 
                        pos, ori,
                        flags=p.URDF_USE_SELF_COLLISION,
                        useFixedBase=False)
        quadruped = self.id

        
        
        for j in range(p.getNumJoints(quadruped)):
            print(p.getJointInfo(quadruped, j))

        #2,5,8 and 11 are the lower legs
        self.lower_legs = [2, 5, 8, 11]
        for l0 in self.lower_legs:
            for l1 in self.lower_legs:
                if (l1 > l0):
                    enableCollision = 1
                    print("collision for pair", l0, l1,
                            p.getJointInfo(quadruped, l0)[12],
                            p.getJointInfo(quadruped, l1)[12], "enabled=", enableCollision)
                    p.setCollisionFilterPair(quadruped, quadruped, 2, 5, enableCollision)

        self.jointIds = []
        paramIds = []
        self.jointOffsets = []
        self.jointDirections = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        jointAngles = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(4):
            self.jointOffsets.append(0)
            self.jointOffsets.append(-0.7)
            self.jointOffsets.append(0.7)

        self.maxForceId = p.addUserDebugParameter("LaikaGo_maxForce", 0, 100, 20)

        for j in range(p.getNumJoints(quadruped)):
            p.changeDynamics(quadruped, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(quadruped, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.jointIds.append(j)

        joints = []

        self.predefined_lines = []
        with open(pd.getDataPath() + "/laikago/data1.txt", "r") as filestream:
            for line in filestream:
                self.predefined_lines.append(line)
            
        self.t = 0
        return

    def step(self):
        line = self.predefined_lines[self.t%len(self.predefined_lines)]
        self.t += 1

        maxForce = p.readUserDebugParameter(self.maxForceId)
        currentline = line.split(",")
        joints = currentline[2:14]
        for j in range(12):
            targetPos = float(joints[j])
            p.setJointMotorControl2(self.id,
                                    self.jointIds[j],
                                    p.POSITION_CONTROL,
                                    self.jointDirections[j] * targetPos + self.jointOffsets[j],
                                    force=maxForce)
        p.stepSimulation()
        for lower_leg in self.lower_legs:
            pts = p.getContactPoints(self.id, -1, lower_leg)
        time.sleep(1. / 500.)
        return