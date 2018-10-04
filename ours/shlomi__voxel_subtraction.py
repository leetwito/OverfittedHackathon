
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from time import time
from tqdm import tqdm

from utilities.math_utils import RotationTranslationData
from utilities import data_utils
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from glob import glob


# In[2]:


def pc_to_grid(df, voxel_size, dest_dir, dest_name):
    # df_frame points are in world coordinate system
    # dest_name may be frame number
    start_time = time()
#     df = pd.read_csv(csv_path, header=None)

    df_xyz = df.iloc[:,:3]
    ref_ser = df.iloc[:, 3]
    df_grid = df_xyz//voxel_size
    df_grid[3] = ref_ser
    df_grid.columns=list("xyzr")
    df_voxel = df_grid.groupby(['x','y','z']).apply(lambda x: x.iloc[:, 3].mean()).to_frame()
    df_voxel.reset_index(drop=False, inplace=True)
    df_voxel.iloc[:, :3] = df_voxel.iloc[:, :3]*voxel_size + voxel_size//2
    df_voxel= df_voxel.astype(np.int)
    base_name = os.path.join(dest_dir, dest_name)
    df_voxel.to_csv(base_name + '_pointcloud.csv', header=None, index=False)
    pd.DataFrame([0]*df_voxel.shape[0]).to_csv(base_name+'_labels.csv', header=None, index=False)
    pd.DataFrame([0.]*6).T.to_csv(base_name+'_egomotion.csv', header=None, index=False)
    print('single file runtime: {}'.format(time()-start_time))


# In[3]:


world_dir = "E:/Datasets/DataHack/World/Train/vid_1/"


# In[4]:


scene_frames = glob(world_dir+"/*point*")[:10] + glob(world_dir+"/*point*")[20:30]
cur_frame = glob(world_dir+"/*point*")[15]


# In[5]:


s_pcs = []
for f in scene_frames:
    s_pcs.append(pd.read_csv(f, header=None))
df_scene_pcs = pd.concat(s_pcs, axis=0)


# In[6]:


get_ipython().run_cell_magic('time', '', 'try:\n    os.mkdir("tmp/")\nexcept:\n    pass\npc_to_grid(df_scene_pcs, 20, "tmp", "0000000")')


# In[7]:


get_ipython().run_cell_magic('time', '', 'df_cur_pc = pd.read_csv(cur_frame, header=None)\npc_to_grid(df_cur_pc, 20, "tmp", "0000001")')


# In[8]:


w = pd.read_csv("tmp/0000000_pointcloud.csv", header=None)
c = pd.read_csv("tmp/0000001_pointcloud.csv", header=None)


# In[9]:


w['voxel_id'] = w.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
c['voxel_id'] = c.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)


# In[10]:


c_out = c[~c.voxel_id.isin(w.voxel_id)]


# In[11]:


c.shape, c_out.shape


# In[12]:


c_out = c_out.iloc[:, :4]
c_out.head()


# In[16]:


try:
    os.mkdir("tmp_out/")
except:
    pass
pc_to_grid(c_out, 20, "tmp_out", "0000002")

