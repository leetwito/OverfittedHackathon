
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import open3d
# examples/Python/Tutorial/Basic/icp_registration.py
    
from open3d import *
import numpy as np
import copy


# In[18]:


path = "voxelling_output/submission_files/check_registration/"
pcw1 = pd.read_csv(path+"0000000_pointcloud.csv", header=None)
pcw2 = pd.read_csv(path+"0000150_pointcloud.csv", header=None)

pcw1.iloc[:, :3].to_csv(path+"source.xyz", sep=" ", header=None, index=None)
pcw2.iloc[:, :3].to_csv(path+"target.xyz", sep=" ", header=None, index=None)


# In[19]:


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])


# In[20]:


source = read_point_cloud(path+"source.xyz")
target = read_point_cloud(path+"target.xyz")


# In[21]:


source


# In[72]:


threshold = 1000000000000000000

# trans_init = np.eye(4)
# trans_init[:3, 3] = 100*np.random.rand(3)

trans_init = np.array([[np.cos(np.pi/4), np.sin(np.pi/4), 0., 10.],
                      [-np.sin(np.pi/4), np.cos(np.pi/4), 0., 20.],
                      [0,0,1,10],
                      [0,0,0,1]])
trans_init


# In[73]:


# draw_registration_result(source, target, trans_init)


# In[74]:


print("Initial alignment")
evaluation = evaluate_registration(source, target,
        threshold, trans_init)
print(evaluation)


# In[75]:


print("Apply point-to-point ICP")
reg_p2p = registration_icp(source, target, threshold, trans_init,
        TransformationEstimationPointToPoint(), )
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
print("")
draw_registration_result(source, target, reg_p2p.transformation)


# In[63]:


print("Apply point-to-plane ICP")
reg_p2l = registration_icp(source, target, threshold, trans_init,
        TransformationEstimationPointToPlane())
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
print("")
draw_registration_result(source, target, reg_p2l.transformation)


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

