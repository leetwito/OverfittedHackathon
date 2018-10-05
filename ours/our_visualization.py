
# coding: utf-8

# In[1]:


import pandas as pd
from visualizations.vis import pcshow
import os.path as osp
import numpy as np

import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


idx = 33
filename = '0'*(7-len(str(idx)))+str(idx)+"_labels.csv"
filename


# In[3]:


pcw = pd.read_csv("voxelling_output/World2/Train/vid_1/"+filename.replace("labels", "pointcloud"), header=None)/100
pred = pd.read_csv("voxelling_output/submission_files/vid_1_pred/"+filename, header=None)
gt = pd.read_csv("voxelling_output/submission_files/vid_1_gt/"+filename, header=None)


# In[4]:


# labeled_pc
pcw["gt"] = gt.values
pcw = pcw.values


# In[5]:


mask = np.zeros(pcw.shape[0]).astype(int)
mask[(pred==1.).values.T[0] & (pred==gt).values.T[0]] = 1       # tp - purple
mask[(pred==1.).values.T[0] & (pred!=gt).values.T[0]] = 2       # fp - orange
mask[(pred==0.).values.T[0] & (pred!=gt).values.T[0]] = 3       # fn - green


# In[6]:


pcshow(pcw, max_points=80000, point_cloud_coloring=mask)


# In[7]:


df = pd.DataFrame(pcw)


# In[8]:


df.columns = list('xyzrl')


# In[9]:


# df[["x", "z"]].plot(kind="scatter", x="x", y="z")


# In[10]:


df["z_mod"] = df.z/df.x#/np.tan(0.2*np.pi/180)


# In[11]:


df.plot(kind="scatter", x="x", y="z_mod")

