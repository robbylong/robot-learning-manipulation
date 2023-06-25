import pybullet as p

class Robot (object):
    def __init__(self, path, pos, ori, **kwargs):  
        # Simulates URDF from given path at given
        # position/orientation
        # path : string : relative or absolute path to URDF/Xacro
        #                 note, PyBullet needs reference-less URDF
        # pos : [x,y,z] list-like position
        # ori : [x,y,z,w] list-like quaternion 
        self.id = p.loadURDF(path, pos, ori, **kwargs)
        self.init_pos = pos
        self.init_ori = ori
        pass
    def step (self):
        # Things that need to be done in every step
        # of simulation 
        pass