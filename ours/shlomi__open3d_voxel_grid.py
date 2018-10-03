
# coding: utf-8

# In[1]:


# install open3d by: conda install -c open3d-admin open3d
import open3d
import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("data_examples/test_video/0000000_pointcloud.csv", header=None)
df.iloc[:, :3].to_csv("data_examples/test_video_mod/0000000_pointcloud.xyz", header=None, index=False, sep=" ")


# In[3]:


# for creating xyzrgb format (r=intensity, g,b=0)
df2 = df.copy()
df2[3] = df2[3]/100
df2[4] = 0
df2[5] = 0

df2.to_csv("data_examples/test_video_mod/0000000_pointcloud.xyzrgb", header=None, index=False, sep=" ")
df2.head()


# In[4]:


pcd = open3d.read_point_cloud("data_examples/test_video_mod/0000000_pointcloud.xyzrgb")
pcd_pts = np.asarray(pcd.points)


# In[5]:


print(pcd)


# In[6]:


print(pcd_pts)


# In[7]:


# open3d.draw_geometries([pcd])


# In[8]:


downpcd = open3d.voxel_down_sample(pcd, voxel_size = 0.00001)
downpcd_pts = np.asarray(downpcd.points)


# In[9]:


print(downpcd_pts)


# In[10]:


# open3d.draw_geometries([downpcd])


# In[11]:


df_pcd = pd.DataFrame(pcd_pts)
print("min points:")
print(df_pcd.min(axis=0))
print("\nmax points:")
print(df_pcd.max(axis=0))
# df_pcd.iloc[:, 0].hist()


# In[12]:


df_dwn = pd.DataFrame(downpcd_pts).astype(int)
print("min points:")
print(df_dwn.min(axis=0))
print("\nmax points:")
print(df_dwn.max(axis=0))


# In[13]:


pcd_pts.shape, downpcd_pts.shape
# some points were merged to same voxel.

