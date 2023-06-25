import pickle
import numpy as np
import torch

# insert the path to the 'networks.py' file
import sys
sys.path.insert(0, "/home/xiang/RPL-affcorrs-gripper/src/")
import networks

# open the policy file (which has been stored with pickle)
filepath = "/home/xiang/RPL-affcorrs-gripper/src/"
filename = "DQN_150x100x50_policy_net_001.pickle"
print(f"Trying to load the file: {filepath + filename}")
with open(filepath + filename, 'rb') as f:
  loaded_network = pickle.load(f)

n_inputs = 59
n_outputs = 8

# what are the actions? X=gripper prismatic joint, Y=gripper revolute joint, Z=gripper palm, H=gripper height
action_names = ["X_close", "X_open", "Y_close", "Y_open", "Z_close", "Z_open", "H_down", "H_up"]

# # lets do an example state vector
# state = 2 * np.random.rand((n_inputs)) - 1                   # get random vector of numbers from [-1, 1]
# state = np.array([state])                                    # state vectors must be [nested] once and must be Floats
# state_tensor = torch.tensor(state, dtype=torch.float32)      # convert to pytorch and change double->float


class ExplorePolicyNet():
  def __init__(self):
    pass
  
  def run_policy(self, state):
    state = np.array([state])
    state_tensor = torch.tensor(state,dtype=torch.float32)
    # print("Our random input state tensor is:\n", state_tensor)

    # disable gradients as we are not training
    with torch.no_grad():
      # t.max(1) returns largest column value of each row
      # [1] is second column of max result, the index of max element
      # view(1, 1) selects this element which has max expected reward
      action = loaded_network(state_tensor).max(1)[1].view(1, 1)

    # extract the chosen action, which are numbered 0-7
    print(f"Action number is: {action.item()}, this means {action_names[action]}")
    return action.item()