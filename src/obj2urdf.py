import pybullet as p
import pybullet_data

# Set the paths to the OBJ and output URDF files
obj_file = "src/object_models/model.obj"
urdf_file = "src/object_models/mug.urdf"

# Start the PyBullet simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the OBJ file as a URDF
obj_id = p.loadURDF(obj_file, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)

# Save the loaded object as a URDF file
p.saveBullet(urdf_file)

# Disconnect from the PyBullet simulation
p.disconnect()


