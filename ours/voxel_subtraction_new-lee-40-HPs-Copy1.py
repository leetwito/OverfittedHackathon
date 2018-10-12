
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
import pickle

from time import time
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Params and Paths:

# In[ ]:


const_n_frames_per_side=100
const_shift=150

rot_n_frames_per_side=8
rot_shift=5

voxel_size=20
voting_scene_frac_to_drop = 0.3


# In[ ]:


base_dir = "C:/Users\shlomi\Documents\Work\OverfittedHackathon\ours/voxelling_output\Test_world_2/vid_40/"
pickles_path = "voxelling_output/"
orig_files_path = "C:/Users\shlomi\Documents\Work\OverfittedHackathon/ours/voxelling_output/Test/vid_40/"
labels_path = "voxelling_output/submission_files/test_vid_40_pred__final/"
# labels_path = "voxelling_output/submission_files/checks/"


# -----------------------------

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


all_files = glob(base_dir+"/*point*")


# In[ ]:


vid = base_dir.split("/")[-2]


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


list_df_file = pickles_path+"list_df_voxel_scene__%s__voxel_size_%d.p"%(vid, voxel_size)

if os.path.exists(list_df_file):
    list_df_voxel_scene = pickle.load(open(list_df_file, "rb"))
else:
    list_df_voxel_scene = create_list_dfs_voxel_scene(base_dir, voxel_size, upto=None)
    pickle.dump(list_df_voxel_scene, open(pickles_path+"list_df_voxel_scene__%s__voxel_size_%d.p"%(vid, voxel_size), "wb"))


# ----------------------

# In[ ]:


def get_list_idx_for_frame(frame_idx, n_frames_per_side, shift):
    list_idx = np.array(list(np.arange(frame_idx-shift-n_frames_per_side,frame_idx-shift)) + list(np.arange(frame_idx+shift+1,frame_idx+shift+n_frames_per_side+1)))
    list_idx = [i for i in list_idx if i>=0 and i<=list_df_voxel_scene.shape[0]]
    return list_idx


# In[ ]:


def create_scene_voxel_df(list_df_voxel_scene, list_idx, frac_to_drop=0.5):
#     print(list_idx)
    list_df_voxel_scene_for_frame = list_df_voxel_scene[list_idx]
    df_scene = pd.concat(list_df_voxel_scene_for_frame)#.drop_duplicates("voxel_id")
#     print(df_scene)
#     df_scene.keys()
#     df_scene.groupby(["x_grid", "y_grid", "z_grid"]).value_counts()
    dfff = df_scene.groupby(["x_grid", "y_grid", "z_grid"]).apply(len).reset_index(drop=False)

    dfff = dfff[dfff[0]>frac_to_drop*len(list_idx)]
    dfff["v_index"] = dfff.iloc[:, :3].apply(lambda x: hash((x.iloc[0], x.iloc[1], x.iloc[2])), axis=1)
#     print(df_scene.shape)
    df_scene.drop_duplicates(["x_grid", "y_grid", "z_grid"], inplace=True)
#     print(df_scene.shape)
    df_scene = df_scene[df_scene.voxel_id.isin(dfff.v_index)]
#     print(df_scene.shape)
    
    return df_scene


# In[ ]:


def get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift):
    
    list_idx_uf = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
#     print ('--get_df_voxel_sub--\nlist idx_uf: {}\nlist_idx   :{}\n-------'.format(list_idx_uf, list_idx))
    df_voxel_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx, frac_to_drop=voting_scene_frac_to_drop)
    df_voxel_frame = create_scene_voxel_df(list_df_voxel_scene, [frame_idx])

    df_subtracted_frame = df_voxel_frame[~df_voxel_frame.voxel_id.isin(df_voxel_scene.voxel_id)]
    return df_subtracted_frame


# In[ ]:


def point_to_voxel(ser, voxel_size):
    # ser: [x, y, z, r]
    ser_out = ser.iloc[:3]//voxel_size
    ser_out = ser_out*voxel_size + voxel_size//2
    voxel_id = hash((ser_out[0], ser_out[1], ser_out[2]))
    
#     return pd.Series(ser_out.tolist()+[voxel_id])
    return voxel_id


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


# #### Filters:

# In[ ]:


def remove_distant_lines(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

#     dff = dff[dff.x>dff.x.min()+(dff.x.max()-dff.x.min())*0.2]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

    dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[ ]:


def remove_distant_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 
#     print(dff.describe())
    dff = dff[dff["x"]>5500]
#     print(dff.describe())
#     rr = np.sqrt(dff.x**2+dff.y**2)
#     dff = dff[rr<10]
#     print(dff.shape)
#     dff.head()
    
#     dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[ ]:


def remove_high_theta_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

    theta = np.abs(np.arctan(dff.y/dff.x))
    dff = dff[theta>0.9*theta.max()]
    dff = dff[dff.x>300]
#     print(dff.shape)
#     dff.head()
    
#     dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.3]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
#     print(dff.shape)
#     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[ ]:


def remove_close_lines(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

    dff = dff[dff.x<dff.x.min()+(dff.x.max()-dff.x.min())*0.1]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z<dff.z.min()+(dff.z.max()-dff.z.min())*0.01]
#     print(dff.shape)
#     dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# #     print(dff.shape)
# #     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[ ]:


def remove_high_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

#     dff = dff[dff.x<dff.x.min()+(dff.x.max()-dff.x.min())*0.15]
#     print(dff.shape)
#     dff.head()
    
    dff = dff[dff.z>400]
    print(dff.shape)
    dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# #     print(dff.shape)
# #     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# In[ ]:


def remove_low_points(frame_idx, df_labels, orig_files_path):
    filename = (7-len(str(frame_idx)))*"0"+str(frame_idx)+"_pointcloud.csv"
    dff = pd.read_csv(orig_files_path+filename, header=None)
    dff.columns = list("xyzr")
    dff["l"] = df_labels.tolist()
    dff = dff[dff.l==1]
#     print(dff.shape)
#     dff.head() 

#     dff = dff[dff.x<dff.x.min()+(dff.x.max()-dff.x.min())*0.15]
#     print(dff.shape)
#     dff.head()
#     print("\n\nXMIN:", dff.x.min())
    dff = dff[dff.z<dff.z.min()+15]
    print(dff.shape)
    dff.head()

#     dff = dff[(dff["r"]>5)&(dff["r"]<15)]
# #     print(dff.shape)
# #     dff.head()

    df_labels.loc[dff.index] = False
    return df_labels


# ---------------------------------

# In[ ]:


# frame_idx=15
frames_range = range(1000)
# frames_range = range(15,900)
for frame_idx in tqdm(frames_range):
    print("=========================================================================")
    print("frame_idx = %d"%frame_idx)
    print("=========================================================================")

    tic = time()
    
    shift = const_shift
    n_frames_per_side = const_n_frames_per_side
    
    list_idx_uf = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
    list_idx = [i for i in list_idx_uf if i>=0 and i<=list_df_voxel_scene.shape[0]-1]
#     print ('--main--\nlist idx_uf: {}\nlist_idx   :{}\n-------'.format(list_idx_uf, list_idx))
    toc = time(); print(toc-tic, ": list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)"); tic=time()
    
    df_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx)
    
    
#     if frame_idx%frames_per_scene==0 :
    df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)
    toc = time(); print(toc-tic, ": df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)"); tic=time()
    
    df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)
#     df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)
    df_frame_orig["voxel_id"] =  ((df_frame_orig.iloc[:, :3]//voxel_size)*voxel_size + voxel_size//2).apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
    toc = time(); print(toc-tic, ": df_frame_orig['voxel_id'] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)"); tic=time()
    
    df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
    toc = time(); print(toc-tic, ": df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)"); tic=time()
    
    if (df_labels.sum()/df_labels.shape[0])>0.5:
        print("\nIN ROTATION MODE! fraction of green pixels before:", (df_labels.sum()/df_labels.shape[0]))
        n_frames_per_side=rot_n_frames_per_side
        shift=rot_shift
        
        list_idx_uf = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)
        list_idx = [i for i in list_idx_uf if i>=0 and i<=list_df_voxel_scene.shape[0]-1]
        print("\nIN ROTATION MODE: list_idx:", list_idx)
    #     print ('--main--\nlist idx_uf: {}\nlist_idx   :{}\n-------'.format(list_idx_uf, list_idx))
        toc = time(); print(toc-tic, ": list_idx = get_list_idx_for_frame(frame_idx, n_frames_per_side, shift)"); tic=time()

        df_scene = create_scene_voxel_df(list_df_voxel_scene, list_idx)


    #     if frame_idx%frames_per_scene==0 :
        df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)
        toc = time(); print(toc-tic, ": df_subtracted_frame = get_df_voxel_subtracted_frame(list_df_voxel_scene, frame_idx, n_frames_per_side, shift)"); tic=time()

        df_frame_orig = pd.read_csv(all_files[frame_idx], header=None)
    #     df_frame_orig["voxel_id"] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)
        df_frame_orig["voxel_id"] =  ((df_frame_orig.iloc[:, :3]//voxel_size)*voxel_size + voxel_size//2).apply(lambda x: hash((x[0], x[1], x[2])), axis=1)
        toc = time(); print(toc-tic, ": df_frame_orig['voxel_id'] =  df_frame_orig.apply(lambda x: point_to_voxel(x, voxel_size), axis=1)"); tic=time()

        df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)
        toc = time(); print(toc-tic, ": df_labels = df_frame_orig.voxel_id.isin(df_subtracted_frame.voxel_id)"); tic=time()

        print("\nIN ROTATION MODE! fraction of green pixels after:", (df_labels.sum()/df_labels.shape[0]))
        
    
    
    # filters
    df_labels = remove_distant_lines(frame_idx, df_labels, orig_files_path)
    df_labels = remove_distant_points(frame_idx, df_labels, orig_files_path)
    df_labels = remove_close_lines(frame_idx, df_labels, orig_files_path)
    df_labels = remove_high_points(frame_idx, df_labels, orig_files_path)
    df_labels = remove_low_points(frame_idx, df_labels, orig_files_path)
    df_labels = remove_high_theta_points(frame_idx, df_labels, orig_files_path)
    
#     df_labels.iloc[:] = False
    
    df_frame_orig_subtracted = df_frame_orig[df_labels]
    toc = time(); print(toc-tic, ": df_frame_orig_subtracted = df_frame_orig[df_labels]"); tic=time()
    
    print(df_frame_orig_subtracted.shape)
    df_labels = df_labels.astype(int)
    file_labels = labels_path + os.path.basename(all_files[frame_idx]).replace("pointcloud", "labels")
    print("file_labels:", file_labels)
    df_labels.to_csv(file_labels, header=None, index=False)
    toc = time(); print(toc-tic, ': df_labels.to_csv(pickles_path+"df_labels__frame_%d__%s__voxel_size_%d_(%d,%d,%d).p"%(frame_idx, vid, voxel_size, n_frames_per_side, shift, n_frames_per_side))'); tic=time()
    
    print(df_labels.sum())
    df_frame_orig_subtracted = df_frame_orig_subtracted.iloc[:, :4]
#     df_frame_orig_subtracted.to_csv(pickles_path+"df_frame_orig_subtracted__frame_%d__%s__voxel_size%d_(%d,%d,%d).p"%(frame_idx, vid, voxel_size, n_frames_per_side, shift, n_frames_per_side)), header=None, index=False)
#     save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)
    toc = time(); print(toc-tic, ': save_frame_for_movie(df_frame_orig_subtracted, "tmp_only_labeled/", all_files, frame_idx)'); tic=time()


# In[ ]:


list_idx

