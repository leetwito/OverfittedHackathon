
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from time import time
from tqdm import tqdm
from utilities.math_utils import RotationTranslationData
from utilities import data_utils
from glob import glob


# In[ ]:


def add_grid_and_hash(df, voxel_size):
    """ 
    add grid and hash to each point of a given point cloud
    """
    df_grid = df.iloc[:,:3]//voxel_size
    df_grid.iloc[:, :3] = df_grid.iloc[:, :3]*voxel_size + voxel_size//2
    df_grid['voxel_id'] = df_grid.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
    df_grid = pd.concat([df, df_grid], axis=1)
    df_grid.columns = ['x', 'y', 'z', 'r', 'x_grid', 'y_grid', 'z_grid', 'voxel_id']
    return df_grid


# In[ ]:


# def classify(video_dir, scene_size, freeze_size, voxel_size):
#     """ scene_size - num of frames befor and after the the examed frame
#         freeze_size - num of frames to skip
#     """
#     befor = scene_size
#     after = freeze_size + scene_size
#     end = 2*scene_size + freeze
    
#     scene_frames = glob(video_dir+"/*point*")[:befor] + glob(video_dir+"/*point*")[after:end]
    


# In[ ]:


# %%time
# base_dir = "E:/Datasets/DataHack/World/Train/vid_1/"
# list_df_voxel_scene = []
# for f in glob(base_dir+"/*point*")[:31]:
#     list_df_voxel_scene.append( add_grid_and_hash(pd.read_csv(f, header=None), 20))


# In[ ]:


# len(list_df_voxel_scene)
# list_df_voxel_scene[0]


# -----------------------------

# In[ ]:


base_dir = "E:/Datasets/DataHack/World/Train/vid_1/"
n_frames_per_side=10
shift=5
voxel_size=20


# In[ ]:


all_files = glob(base_dir+"/*point*")


# In[ ]:


def create_list_dfs_voxel_scene(base_dir, voxel_size, upto=None):
    global all_files
    list_df_voxel_scene = []
#     all_files = glob(base_dir+"/*point*")
    if upto is not None:
        all_files = all_files[:upto]
    
    for f in tqdm(all_files):
        list_df_voxel_scene.append( add_grid_and_hash(pd.read_csv(f, header=None), voxel_size))
    list_df_voxel_scene = pd.Series(list_df_voxel_scene).values
    return list_df_voxel_scene


# In[ ]:


list_df_voxel_scene = create_list_dfs_voxel_scene(base_dir, voxel_size, upto=50)


# ----------------------

# In[ ]:


# frame_idx=15
# todo: add skips


# In[ ]:


def get_list_idx_for_frame(frame_idx, n_frames_per_side, shift):
    list_idx = np.array(list(np.arange(frame_idx-shift-n_frames_per_side,frame_idx-shift)) + list(np.arange(frame_idx+shift+1,frame_idx+shift+n_frames_per_side+1)))
    return list_idx


# In[ ]:


# list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
# list_idx


# In[ ]:


def create_scene_voxel_df(list_df_voxel_scene, list_idx):
    list_df_voxel_scene_for_frame = list_df_voxel_scene[list_idx]
    df_scene = pd.concat(list_df_voxel_scene_for_frame).drop_duplicates("voxel_id")
    return df_scene


# In[ ]:


def get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift):
    
    list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
    
    df_voxel_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx)
    df_voxel_frame = create_scene_voxel_df(list_df_voxel_scene, [frame_idx])

    df_subtracted_frame = df_voxel_frame[~df_voxel_frame.voxel_id.isin(df_voxel_scene.voxel_id)]
    return df_subtracted_frame


# In[ ]:


# df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)


# In[ ]:


# df_subtracted_frame.head()


# In[ ]:


def point_to_voxel(ser, voxel_size):
    # ser: [x, y, z, r]
    ser_out = ser.iloc[:3]//voxel_size
    ser_out = ser_out*voxel_size + voxel_size//2
    voxel_id = hash((ser_out[0], ser_out[1], ser_out[2]))
    
#     return pd.Series(ser_out.tolist()+[voxel_id])
    return voxel_id


# In[ ]:


# df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)


# In[ ]:


# %%time
# df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)


# In[ ]:


# df_frame_orig.head()


# In[ ]:


# df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
# df_labels.sum()


# In[ ]:


# df_frame_orig_subtracted = df_frame_orig[df_labels]
# df_labels = df_labels.astype(int)


# In[ ]:


# df_frame_orig_subtracted.shape


# In[ ]:


# df_frame_orig_subtracted = df_frame_orig_subtracted.iloc[:4]


# In[ ]:


# # 
# save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)


# In[ ]:


def save_frame_for_movie(frame, folder, all_files, frame_idx):
    if not os.path.exists(folder):
        os.mkdir(folder)
    filename = os.path.basename(all_files[frame_idx])
    frame.to_csv(folder + filename, header=None, index=False)
    labels_filename = filename.replace('pointcloud', 'labels')
    pd.DataFrame([0]*frame.shape[0]).to_csv(folder + labels_filename, header=None, index=False)

    egomotion_filename = filename.replace('pointcloud', 'egomotion')
    pd.DataFrame([0.]*6).T.to_csv(folder + egomotion_filename, header=None, index=False)
    print('frame {} saved successfuly'.format(frame_idx))
    


# In[ ]:


# frame_idx=15
for frame_idx in tqdm(range(15,25)):
    list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
    df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)
    df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)
    df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)
    df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
    df_frame_orig_subtracted = df_frame_orig[df_labels]
    print(df_frame_orig_subtracted.shape)
    df_labels = df_labels.astype(int)
    print(df_labels.sum())
    df_frame_orig_subtracted = df_frame_orig_subtracted.iloc[:4]
    save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)


# In[ ]:


# base_dir = "E:/Datasets/DataHack/Train"
# for sub_dir in os.listdir(base_dir):
#     print(sub_dir)
# #     for f in glob(sub_dir+"/*point*"):
# #     df = add_grid_and_hash(pd.read_csv(f), header=None), 20)


# In[ ]:


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
    
    return df_voxel


# In[ ]:


world_dir = "E:/Datasets/DataHack/World/Train/vid_1/"


# In[ ]:


scene_frames = glob(world_dir+"/*point*")[:10] + glob(world_dir+"/*point*")[20:30]
cur_frame = glob(world_dir+"/*point*")[15]


# In[ ]:


s_pcs = []
for f in scene_frames:
    s_pcs.append(pd.read_csv(f, header=None))
df_scene_pcs = pd.concat(s_pcs, axis=0)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'try:\n    os.mkdir("tmp/")\nexcept:\n    pass\npc_to_grid(df_scene_pcs, 20, "tmp", "0000000")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_cur_pc = pd.read_csv(cur_frame, header=None)\npc_to_grid(df_cur_pc, 20, "tmp", "0000001")')


# In[ ]:


w = pd.read_csv("tmp/0000000_pointcloud.csv", header=None)
c = pd.read_csv("tmp/0000001_pointcloud.csv", header=None)


# In[ ]:


w['voxel_id'] = w.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
c['voxel_id'] = c.apply(lambda x: hash((x[0], x[1], x[2])), axis=1)


# In[ ]:


c_out = c[~c.voxel_id.isin(w.voxel_id)]


# In[ ]:


c.shape, c_out.shape


# In[ ]:


c_out = c_out.iloc[:, :4]
c_out.head()


# In[ ]:


try:
    os.mkdir("tmp_out/")
except:
    pass
pc_to_grid(c_out, 20, "tmp_out", "0000002")


# In[ ]:


list_df_voxels = []
for f in g


# In[ ]:


def voxel_classification_for_frame(vid_dir, frame_idx):
    df_pc_scene = ...
    df_pc_frame = ...
    df_voxel_scene = build_voxel(df_pc_scene)
    df_voxel_frame = build_voxel()
    
    
