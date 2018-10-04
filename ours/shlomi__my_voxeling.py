
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from time import time
from tqdm import tqdm


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


df_pc = pd.read_csv('data_examples/test_video/0000000_pointcloud.csv')
voxel_size = 100
dest_dir = 'data_examples/test_video_mod/'
dest_name = '0000000'

pc_to_grid(df_pc, voxel_size, dest_dir, dest_name)


# In[ ]:


from utilities.math_utils import RotationTranslationData
from utilities import data_utils
import os
from tqdm import tqdm
import numpy as np
import pandas as pd


# In[ ]:


def transform_frame_to_world(pc, ego):
    ego_rt = RotationTranslationData(vecs=(ego[:3], ego[3:]))
    ego_pc = ego_rt.apply_transform(pc[:, :3]) # pc is (x, y, z, lumin)
    return ego_pc


# In[ ]:


def transform_folder_to_world(sub_dir):
    base_dir = "E:\Datasets\DataHack\Train"
    base_res_dir = "E:\Datasets\DataHack\World\Train"
    print('Working on sub_dir: {}'.format(sub_dir))

    for idx in data_utils.enumerate_frames(os.path.join(base_dir, sub_dir)):
        pc_file = os.path.join(base_res_dir, data_utils.frame_to_filename(sub_dir, idx, 'pointcloud'))
        if os.path.exists(pc_file):
            continue
        pc, ego, label = data_utils.read_all_data(os.path.join(base_dir, sub_dir), idx)
        ego_pc = transform_frame_to_world(pc, ego)
        ego_pc = np.concatenate((ego_pc, pc[:, 3:4]), -1)
        df = (pd.DataFrame(ego_pc) * 100).astype(int)
        res_dir = os.path.join(base_res_dir, sub_dir)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        file_name = os.path.join(res_dir, str(idx).zfill(7))
        df.to_csv(file_name+'_pointcloud.csv', header=None, index=False)
        pd.DataFrame([0]*df.shape[0]).to_csv(file_name+'_labels.csv', header=None, index=False)
        pd.DataFrame([0.]*6).T.to_csv(file_name+'_egomotion.csv', header=None, index=False)


# In[ ]:


base_dir = "E:\Datasets\DataHack\Train"
transform_folder_to_world('vid_1')


# In[ ]:


# a = pd.read_csv("E:/Datasets/DataHack/Train/vid_1/0000000_pointcloud.csv", header=None)


# In[ ]:


# b = pd.read_csv("E:/Datasets/DataHack/World/Train/vid_1/0000000_pointcloud.csv", header=None)


# In[ ]:


# a.max() - a.min()


# In[ ]:


# b.max() - b.min()

