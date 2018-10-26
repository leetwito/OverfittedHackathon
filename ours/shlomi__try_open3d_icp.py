
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import open3d
# examples/Python/Tutorial/Basic/icp_registration.py
    
from open3d import *
import numpy as np
import copy


# In[2]:


path = "voxelling_output/submission_files/check_registration/"
pcw1 = pd.read_csv(path+"0000007_pointcloud.csv", header=None)
pcw2 = pd.read_csv(path+"0000020_pointcloud.csv", header=None)

pcw1.iloc[:, :3].to_csv(path+"source.xyz", sep=" ", header=None, index=None)
pcw2.iloc[:, :3].to_csv(path+"target.xyz", sep=" ", header=None, index=None)


# In[3]:


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])


# In[4]:


source = read_point_cloud(path+"source.xyz")
target = read_point_cloud(path+"target.xyz")


# In[5]:


source


# In[6]:


threshold = 1000000000000000000

trans_init = np.eye(4)
trans_init[:3, 3] = 100*np.random.rand(3)

# trans_init = np.array([[np.cos(np.pi/8), np.sin(np.pi/8), 0., 10.],
#                       [-np.sin(np.pi/8), np.cos(np.pi/8), 0., 20.],
#                       [0,0,1,10],
#                       [0,0,0,1]])
trans_init


# In[7]:


draw_registration_result(source, target, trans_init)


# In[8]:


print("Initial alignment")
evaluation = evaluate_registration(source, target,
        threshold, trans_init)
print(evaluation)


# In[9]:


print("Apply point-to-point ICP")
reg_p2p = registration_icp(source, target, threshold, trans_init,
        TransformationEstimationPointToPoint(), )
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")
draw_registration_result(source, target, reg_p2p.transformation)


# In[12]:


R = reg_p2p.transformation
print(R)
print(R.shape)


# -------------------------------

# In[35]:


from utilities.math_utils import extract_rotation, extract_translation


# In[36]:


rot = extract_rotation(R)
trans = extract_translation(R)
rot, trans


# ---------------------------------------------

# In[37]:


def rot_and_trans_mat_to_euler_3d(R):
    # based on rotation and translation matrix defenition found here: http://planning.cs.uiuc.edu/node104.html
    assert R[3,3]==1 and R.shape==(4,4) and R[3,1]==0
    x_t = R[0,3]
    y_t = R[1,3]
    z_t = R[2,3]
    beta = -np.arcsin(R[2,0])
    gamma = np.arcsin(R[2,1]/np.cos(beta))
    alpha = np.arcsin(R[1,0]/np.cos(beta))
    return (x_t, y_t, z_t), (alpha, beta, gamma)


# In[38]:


trans, rot = rot_and_trans_mat_to_euler_3d(R)


# In[39]:


rot, trans  ## looks like my function iz good but alpha and gamma should replace positions 


# In[44]:


gt_ego1 = pd.read_csv(path+"0000007_egomotion.csv", header=None)
gt_ego2 = pd.read_csv(path+"0000020_egomotion.csv", header=None)
gt_ego1-gt_ego2


# ----------------------------------------

# In[1]:


from glob import glob
import os
import open3d
from utilities.math_utils import extract_rotation, extract_translation
import numpy as np
import pandas as pd

path = "E:/Datasets/DataHack/Test/vid_21_estimate_egomotion"
trans_init = np.eye(4)
trans_init[:3, 3] = 100*np.random.rand(3)
threshold = 10000

pc_paths = glob(path+'/*pointcloud.csv')
xyz_paths = [i.replace('_pointcloud.csv', '.xyz') for i in pc_paths]

if not os.path.exists(xyz_paths[0]):
    for idx in range(len(pc_paths)):
        pc = pd.read_csv(pc_paths[idx], header=None)
        pc.iloc[:, :3].to_csv(xyz_path[idx], sep=" ", header=None, index=None)

pc_prev = open3d.read_point_cloud(xyz_paths[0])
R_prev = np.eye(4)
cur_ego = pd.DataFrame(np.zeros((1,6)), dtype=np.float)
print(cur_ego)

for xyz_path in xyz_paths[1:]:
    pc_cur = open3d.read_point_cloud(xyz_path)
    evaluation = open3d.evaluate_registration(pc_prev, pc_cur, threshold, trans_init)
    print("Apply point-to-point ICP to: \n{}".format(xyz_path))
    reg_p2p = open3d.registration_icp(pc_prev, pc_cur, threshold, trans_init,
            open3d.TransformationEstimationPointToPoint(), )
    print(reg_p2p)
    print("Transformation is:")
    R_cur = reg_p2p.transformation
    print(R_cur)
    R = np.matmul(R_cur, R_prev)
    rot = extract_rotation(R)
    trans = extract_translation(R)
    cur_ego.at[0, :2] = rot
    cur_ego.at[0, 3:] = trans
    print(cur_ego)
    print("")    
    
    cur_ego.iloc[0, :].T.to_csv(xyz_path.replace('.xyz', '_egomotion.csv'), sep=" ", header=None, index=None)
    R_prev = R
    pc_prev = pc_cur


# -----------------------------

# In[ ]:



from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycpd import rigid_registration
import numpy as np
import time

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def main():
    X = np.loadtxt('data/bunny_target.txt')
    Y = np.loadtxt('data/bunny_source.txt') #synthetic data, equaivalent to X + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    callback = partial(visualize, ax=ax)

    reg = rigid_registration(**{ 'X': X, 'Y': Y })
    reg.register(callback)
    plt.show()

if __name__ == '__main__':
    main()

