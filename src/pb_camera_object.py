import gc
import os
import time
import pybullet as p
import numpy as np
import open3d as o3d

# import pytransform3d.transformations as pytr 
# import pytransform3d.rotations as pyrot 

class Object():
    def __init__(self, obj_path, pos, quat, **kwargs):
        """obj_path points to urdf file of object,
        e.g. from GoogleScanned Objects. 
        
        Pos/quat define the pose in the world.

        self.id of the object can be used to remove it from scene or 
        other pybullet minupulations.
        """
        self.obj_path = obj_path
        flags = p.URDF_USE_INERTIA_FROM_FILE
        if "globalScaling" in list(kwargs.keys()):
            self.id = p.loadURDF(obj_path, pos, 
                                baseOrientation = quat,
                                flags=flags,
                                globalScaling = kwargs["globalScaling"])
            print("successfully loaded")
        else: 
            self.id = p.loadURDF(obj_path, pos, 
                                baseOrientation = quat,
                                flags=flags)
            print("successfully loaded")
    def __int__(self):
        return self.id
    
class SDFObject():
    def __init__(self, obj_path, pos, quat, **kwargs):
        """obj_path points to sdf file of object,
        e.g. from GoogleScanned Objects. 
        
        Pos/quat define the pose in the world.

        self.id of the object can be used to remove it from the scene or 
        perform other pybullet manipulations.
        """
        self.obj_path = obj_path
        flags = p.URDF_USE_INERTIA_FROM_FILE
        if "globalScaling" in list(kwargs.keys()):
            self.id = p.loadSDF(obj_path, globalScaling=kwargs["globalScaling"])[0]
        else: 
            self.id = p.loadSDF(obj_path)[0]
        p.resetBasePositionAndOrientation(self.id, pos, quat)
    
    def __int__(self):
        return self.id



class CameraObject():
    def __init__(self, width=1024, height=1024,
                near = 0.02, far=100, fov = 60,
                pos=[1.5,0,1], target=[1,0,0.8],
                axis = [0, 0, 1]):
        vars = locals() # dict of local names
        self.__dict__.update(vars) # __dict__ holds and object's attributes
        del self.__dict__["self"] # don't need `self`
        # width = 1024
        # height = 1024

        # fov = 60
        # near = 0.02
        # far = 100
        self.aspect = width / height

        self.view_matrix = p.computeViewMatrix(pos, target, axis)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, self.aspect, 
                                                              near, far)

        # Get depth values using the OpenGL renderer
        # images = p.getCameraImage(width, height, view_matrix, projection_matrix, 
        #                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # depth_buffer_opengl = np.reshape(images[3], [width, height])
        # depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    
    def capture(self):
        # Get depth values using Tiny renderer
        images = p.getCameraImage(self.width, 
                                  self.height, 
                                  self.view_matrix, 
                                  self.projection_matrix, 
                                  renderer=p.ER_TINY_RENDERER)
        depth_buffer_tiny = np.reshape(images[3], 
                                [self.width, self.height])
        depth_tiny = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_tiny)
        rgb = images[2].reshape((images[0],images[1],4))
        sem = images[4].reshape((images[0],images[1]))
        return rgb, depth_buffer_tiny, depth_tiny, sem 

    def get_point_cloud(self, depth):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

        # "infinite" depths will have a value close to 1

        # create a 4x4 transform matrix that goes from pixel coordinates 
        # (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        #pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]
        return points.reshape(self.height,self.width,3)
    
    def make_pointcloud(self, vertices, colors):

        # # Convert vertices and colors to NumPy arrays if necessary
        vertices = np.asarray(vertices)
        colors = np.asarray(colors)

        # colors need to be in float range 0-1
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(vertices)

        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.remove_non_finite_points()

        return pcd
    
    def filter_points(self, pcd, x_threshold,y_threshold):

        # Convert the point cloud to a NumPy array
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # Filter out points where the absolute value of x is greater than the threshold
        x_mask = np.abs(points[:, 0]) <= x_threshold

        # Filter out points where the absolute value of y is greater than the threshold
        y_mask = np.abs(points[:, 1]) <= y_threshold

        # Combine the masks using logical AND operation
        mask = np.logical_and(x_mask, y_mask)
        filtered_points = points[mask]
        filtered_colors = colors[mask]
        
        # Create a new point cloud with the filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        
        return filtered_pcd
        
