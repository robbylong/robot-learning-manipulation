readme.txt

The semantic descriptions of the gripper and panda in this folder are generated using xacro from the files in the folder /xacro/semantic/...

This does not happen during roslaunch, so it is possible for these files to be out of date. To manually regenerate them, go the the /xacro/semantic/ directory and type:

$ xacro gripper.semantic.xacro > gripper.semantic.urdf
$ xacro panda.semantic.xacro > panda.semantic.urdf
$ xacro panda_and_gripper.xacro > panda_and_gripper.semantic.urdf

This will create three up to date .urdf files, which can be cut and paste into this folder.
