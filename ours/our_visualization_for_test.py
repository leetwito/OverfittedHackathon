
# coding: utf-8

# In[109]:


import pandas as pd
from visualizations.vis import pcshow
import os.path as osp
import numpy as np

import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


# idx = 185
# filename = '0'*(7-len(str(idx)))+str(idx)+"_labels.csv"
# filename


# In[132]:


filename = "0000900"
pcw = pd.read_csv("voxelling_output/submission_files/checks/%s_pointcloud.csv"%filename, header=None)/100
pred = pd.read_csv("voxelling_output/submission_files/checks/%s_labels.csv"%filename, header=None)
# # gt = pd.read_csv("voxelling_output/submission_files/vid_1_gt/"+filename, header=None)


# In[133]:


# labeled_pc
pcw["gt"] = pred.values
pcw = pcw.values


# In[134]:


mask = np.zeros(pcw.shape[0]).astype(int)
# mask[(pred==1.).values.T[0] & (pred==gt).values.T[0]] = 1       # tp - purple
# mask[(pred==1.).values.T[0] & (pred!=gt).values.T[0]] = 2       # fp - orange
# mask[(pred==0.).values.T[0] & (pred!=gt).values.T[0]] = 3       # fn - green
mask[(pred==1.).values.T[0]] = 3      # tp - purple


# In[135]:


pcshow(pcw, max_points=80000, point_cloud_coloring=mask)


# In[79]:


df = pd.DataFrame(pcw)


# In[80]:


df.columns = list('xyzrl')


# In[47]:


# df[["x", "z"]].plot(kind="scatter", x="x", y="z")


# In[148]:


df["z_mod"] = df.z/df.x#/np.tan(0.2*np.pi/180)


# In[149]:


df.plot(kind="scatter", x="x", y="y")

